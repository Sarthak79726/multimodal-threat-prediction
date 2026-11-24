"""
Stacking Ensemble Pipeline Module.
Implements the end-to-end two-level Stacking Ensemble Fusion Architecture:
Level 0:
    - Base Model 1: Physiological Stress Predictor (P_physio)
    - Base Model 2: Spatial Risk Engine (P_spatial)
    - Base Model 3: Environmental Context Engine (P_env)
Level 1:
    - Meta-Learner: Logistic Regression (beta_0 + sum beta_i P_i)

Includes rigorous dataset validation guards and production provenance gates:
    - Enforces synchronized multimodal schema.
    - Rejects circular label generation (detects if target was derived from weighted late fusion or hazard formulas).
    - Rejects synthetic datasets during production training.
    - Production artifact loading strictly gates against synthetic simulation artifacts.
    - Executes leakage-free OOF training and untouched test set evaluation.
"""

import sys
import json
import pickle
import datetime
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

# Ensure project root is in sys.path
_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import config
from src.fusion.meta_learner import (
    BaseMetaLearner,
    LogisticRegressionMetaLearner,
    RandomForestMetaLearner,
    XGBoostMetaLearner,
    WeightedLateFusionBaseline
)
from src.fusion.oof_generator import OOFGenerator
from src.fusion.evaluation import evaluate_predictions, format_comparison_table, save_evaluation_report

class DataCircularityError(ValueError):
    """Raised when ground truth labels are detected to be derived directly from a formula of input scores."""
    pass

class MissingGroundTruthLabelError(ValueError):
    """Raised when legitimate ground-truth threat outcome labels are missing or single-class."""
    pass

class SyntheticDatasetRejectedError(ValueError):
    """Raised when a synthetic dataset or simulation is submitted for production training."""
    pass

class IneligibleProductionArtifactError(ValueError):
    """Raised when attempting to load a synthetic or unverified artifact into production inference."""
    pass

class StackingEnsemblePipeline:
    """
    Two-Level Stacking Ensemble Fusion Pipeline.
    Level 0: 3 Modality Branches -> [P_physio, P_spatial, P_env]
    Level 1: Meta-Learner (Logistic Regression) -> P_final
    """
    VERSION = "2.0.0-academic"

    REQUIRED_SCHEMA = [
        "sample_id",
        "p_physiological",
        "p_spatial",
        "p_environmental",
        "ground_truth_threat_label"
    ]

    def __init__(
        self,
        meta_learner: Optional[BaseMetaLearner] = None,
        feature_names: Optional[List[str]] = None,
        threshold: float = config.DEFAULT_DECISION_THRESHOLD
    ):
        self.feature_names = feature_names or config.STACKING_META_FEATURES
        self.meta_learner = meta_learner or LogisticRegressionMetaLearner(feature_names=self.feature_names)
        self.threshold = threshold
        self.baseline = WeightedLateFusionBaseline(threshold=self.threshold)
        self.is_trained: bool = False
        self.production_eligible: bool = False
        self.training_metadata: Dict[str, Any] = {}
        self.evaluation_results: Dict[str, Any] = {}

    @classmethod
    def validate_multimodal_dataset(
        cls,
        df: pd.DataFrame,
        target_col: str = "ground_truth_threat_label",
        enforce_production_empirical: bool = False
    ) -> None:
        """
        Validates data compatibility, rejects circular threat labels,
        and optionally enforces production empirical data requirements.
        """
        # 1. Check required feature columns
        for col in ["p_physiological", "p_spatial", "p_environmental"]:
            if col not in df.columns and col not in config.STACKING_META_FEATURES:
                alt_map = {
                    "p_physiological": "physiological_stress_score",
                    "p_spatial": "location_risk_score",
                    "p_environmental": "environment_risk_score"
                }
                if alt_map.get(col) not in df.columns:
                    raise ValueError(f"Missing required meta-feature column '{col}' or '{alt_map.get(col)}' in dataset.")

        # 2. Check target column exists
        if target_col not in df.columns:
            if "threat_label" in df.columns:
                target_col = "threat_label"
            else:
                raise MissingGroundTruthLabelError(
                    f"Ground truth target column '{target_col}' not found in dataset. "
                    "Cannot train a stacking meta-learner without independent target labels."
                )

        y = df[target_col].values
        # 3. Check for valid binary distribution
        unique_classes = np.unique(y[~pd.isna(y)])
        if len(unique_classes) < 2:
            raise MissingGroundTruthLabelError(
                f"Target column '{target_col}' contains only class {unique_classes.tolist()}. "
                "Training requires both positive (threat/risky) and negative (safe) instances."
            )

        # 4. Production Empirical Enforcement Gate
        if enforce_production_empirical:
            # Check dataframe attributes or columns for synthetic markers
            is_synth_attr = (
                getattr(df, "attrs", {}).get("dataset_type") == "synthetic"
                or getattr(df, "attrs", {}).get("dataset_type") == "synthetic_simulation"
            )
            has_synth_col = "is_synthetic" in df.columns and bool(df["is_synthetic"].any())
            
            # Check subject token patterns (e.g. artificial identifiers like Subj_01..Subj_20)
            is_synth_subj = False
            if "subject_id" in df.columns:
                subjs = [str(s) for s in df["subject_id"].dropna().unique()]
                if len(subjs) > 0 and all(s.startswith("Subj_") and s[5:].isdigit() for s in subjs):
                    is_synth_subj = True

            if is_synth_attr or has_synth_col or is_synth_subj:
                raise SyntheticDatasetRejectedError(
                    "Production meta-model training strictly requires an authentic empirical multimodal dataset. "
                    "Synthetic datasets, artificial subject tokens, or mathematical simulation data are rejected for production training."
                )

        # 5. Strict Circularity Detection:
        # Detect whether target was derived directly from the prototype formula:
        # 0.40 * phys + 0.25 * loc + 0.35 * env
        c_phys = "physiological_stress_score" if "physiological_stress_score" in df.columns else "p_physiological"
        c_loc = "location_risk_score" if "location_risk_score" in df.columns else "p_spatial"
        c_env = "environment_risk_score" if "environment_risk_score" in df.columns else "p_environmental"

        weighted_check = (
            0.40 * df[c_phys].values +
            0.25 * df[c_loc].values +
            0.35 * df[c_env].values
        )
        pred_labels = (weighted_check >= 0.50).astype(int)
        match_rate = float(np.mean(pred_labels == y))
        corr = float(np.corrcoef(weighted_check, y)[0, 1]) if len(np.unique(y)) > 1 else 0.0

        if match_rate > 0.95 or corr > 0.95:
            raise DataCircularityError(
                f"Critical Methodological Issue Detected: Ground truth label '{target_col}' matches the "
                f"heuristic weighted sum (0.40*phys + 0.25*loc + 0.35*env) with {match_rate*100:.1f}% accuracy "
                f"(correlation = {corr:.4f}). Training a meta-learner on labels created from the weighted formula "
                "constitutes circular evaluation. Stacking requires independently verified ground-truth labels!"
            )

    def train_with_oof(
        self,
        df_train: pd.DataFrame,
        target_col: str = "threat_label",
        group_col: Optional[str] = "subject_id",
        untouched_test_df: Optional[pd.DataFrame] = None,
        n_splits: int = 5,
        random_state: int = 42,
        enforce_production_empirical: bool = False,
        production_eligible: Optional[bool] = None
    ) -> Dict[str, Any]:
        """
        Executes complete leakage-free Out-Of-Fold meta-learner training.
        """
        # Validate dataset integrity first
        self.validate_multimodal_dataset(
            df_train,
            target_col=target_col,
            enforce_production_empirical=enforce_production_empirical
        )

        # Standardize column mapping
        col_map = {
            "p_physiological": "physiological_stress_score",
            "p_spatial": "location_risk_score",
            "p_environmental": "environment_risk_score"
        }
        df_tr = df_train.copy()
        for k, v in col_map.items():
            if k in df_tr.columns and v not in df_tr.columns:
                df_tr[v] = df_tr[k]

        feature_cols = config.STACKING_META_FEATURES
        X_meta_train = df_tr[feature_cols].copy()
        y_train = df_tr[target_col].values
        groups_train = df_tr[group_col].values if (group_col and group_col in df_tr.columns) else None

        print(f"\n[StackingPipeline] Initiating Leakage-Free OOF Training across {len(df_tr)} samples...", flush=True)

        oof_gen = OOFGenerator(n_splits=n_splits, random_state=random_state)
        cv, cv_name = oof_gen.select_cv_strategy(X_meta_train.values, y_train, groups_train)

        n_samples = len(y_train)
        oof_meta_probs = np.zeros(n_samples, dtype=np.float64)

        splits_iter = cv.split(X_meta_train, y_train, groups_train) if groups_train is not None else cv.split(X_meta_train, y_train)

        for fold_idx, (tr_idx, val_idx) in enumerate(splits_iter):
            X_tr, y_tr = X_meta_train.iloc[tr_idx].values, y_train[tr_idx]
            X_va, y_va = X_meta_train.iloc[val_idx].values, y_train[val_idx]

            fold_stacker = LogisticRegressionMetaLearner(feature_names=feature_cols, random_state=random_state)
            fold_stacker.fit(X_tr, y_tr)
            oof_meta_probs[val_idx] = fold_stacker.predict_proba(X_va)[:, 1]

        oof_metrics = evaluate_predictions(y_train, oof_meta_probs, threshold=self.threshold, model_name="Stacking_OOF")
        print(f"[StackingPipeline] OOF Validation F1: {oof_metrics['f1_score']:.4f} | ROC-AUC: {oof_metrics['roc_auc']} | Brier: {oof_metrics['brier_score']:.4f}", flush=True)

        # Retrain final Level-1 Meta-Learner on full training matrix
        self.meta_learner.fit(X_meta_train.values, y_train)
        self.is_trained = True
        self.production_eligible = (
            production_eligible if production_eligible is not None else enforce_production_empirical
        )

        self.training_metadata = {
            "pipeline_version": self.VERSION,
            "trained_at": datetime.datetime.now().isoformat(),
            "cv_strategy": cv_name,
            "n_splits": n_splits,
            "n_samples_train": n_samples,
            "number_of_samples": n_samples,
            "number_of_subjects": len(np.unique(groups_train)) if groups_train is not None else 1,
            "target_column": target_col,
            "group_column": group_col,
            "feature_names": feature_cols,
            "validation_strategy": cv_name,
            "code_version": self.VERSION,
            "production_eligible": self.production_eligible,
            "dataset_type": "empirical_multimodal" if self.production_eligible else "synthetic_simulation",
            "meta_learner_name": self.meta_learner.model_name,
            "meta_learner_explain": self.meta_learner.explain(),
            "oof_validation_metrics": oof_metrics
        }

        if untouched_test_df is not None:
            print(f"[StackingPipeline] Evaluating on Untouched Test Set ({len(untouched_test_df)} samples)...", flush=True)
            df_te = untouched_test_df.copy()
            for k, v in col_map.items():
                if k in df_te.columns and v not in df_te.columns:
                    df_te[v] = df_te[k]

            X_meta_test = df_te[feature_cols].values
            y_test = df_te[target_col].values

            test_meta_probs = self.meta_learner.predict_proba(X_meta_test)[:, 1]
            test_metrics = evaluate_predictions(y_test, test_meta_probs, threshold=self.threshold, model_name="Stacking_Test")
            print(f"[StackingPipeline] Untouched Test F1: {test_metrics['f1_score']:.4f} | ROC-AUC: {test_metrics['roc_auc']} | Brier: {test_metrics['brier_score']:.4f}", flush=True)

            self.training_metadata["untouched_test_metrics"] = test_metrics

        return self.training_metadata

    def predict(
        self,
        p_physio: float,
        p_spatial: float,
        p_env: float
    ) -> Dict[str, Any]:
        """
        Unified inference executing Level 1 Meta-Learner (or baseline fallback).
        """
        vec = np.array([[float(p_physio), float(p_spatial), float(p_env)]], dtype=np.float64)

        # Baseline evaluation
        baseline_score = self.baseline.predict_score(p_physio, p_spatial, p_env)
        baseline_status = "RISKY" if baseline_score >= self.threshold else "SAFE"

        # Check if meta-learner has been trained
        is_trained = (
            self.is_trained
            and getattr(self.meta_learner, "is_trained_ml", False)
        )

        if is_trained:
            meta_prob = float(self.meta_learner.predict_proba(vec)[0, 1])
            meta_status = "RISKY" if meta_prob >= self.threshold else "SAFE"
            meta_explanation = self.meta_learner.explain(sample_vec=vec[0])
            active_score = meta_prob
            active_status = meta_status
            model_used = self.meta_learner.model_name
            meta_model_status = (
                "TRAINED (Production-Eligible)"
                if getattr(self, "production_eligible", False)
                else "TRAINED (Synthetic Simulation / Demo)"
            )
        else:
            meta_prob = None
            meta_status = None
            meta_explanation = None
            active_score = baseline_score
            active_status = baseline_status
            model_used = "WeightedLateFusionBaseline"
            meta_model_status = "UNTRAINED — VALID MULTIMODAL THREAT DATASET REQUIRED"

        return {
            "p_physiological": float(p_physio),
            "p_spatial": float(p_spatial),
            "p_environmental": float(p_env),
            "active_threat_probability": active_score,
            "active_threat_score_pct": active_score * 100.0,
            "active_threat_status": active_status,
            "model_used": model_used,
            "is_meta_learner_trained": is_trained,
            "production_eligible": getattr(self, "production_eligible", False),
            "meta_model_status": meta_model_status,
            "stacking_meta_probability": meta_prob,
            "stacking_meta_status": meta_status,
            "stacking_explainability": meta_explanation,
            "baseline_score": baseline_score,
            "baseline_score_pct": baseline_score * 100.0,
            "baseline_status": baseline_status,
            "threshold_used": self.threshold
        }

    def save_artifact(
        self,
        output_path: Path = config.MODELS_DIR / "fusion" / "stacking_ensemble.pkl",
        dataset_source: str = "Empirical Verified Multimodal Safety Records",
        dataset_type: Optional[str] = None,
        target_definition: str = "Independent verified safety threat event",
        target_provenance: str = "Verified physical incident records",
        production_eligible: Optional[bool] = None
    ) -> None:
        """Serializes pipeline artifact with full academic provenance metadata."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if production_eligible is None:
            production_eligible = getattr(self, "production_eligible", False)

        if dataset_type is None:
            dataset_type = "empirical_multimodal" if production_eligible else "synthetic_simulation"

        # Consolidate metadata
        meta = self.training_metadata.copy()
        meta.update({
            "dataset_source": dataset_source,
            "dataset_type": dataset_type,
            "target_definition": target_definition,
            "target_provenance": target_provenance,
            "production_eligible": production_eligible,
            "training_timestamp": datetime.datetime.now().isoformat(),
            "code_version": self.VERSION,
            "feature_names": self.feature_names
        })

        artifact = {
            "version": self.VERSION,
            "meta_learner": self.meta_learner,
            "baseline": self.baseline,
            "feature_names": self.feature_names,
            "threshold": self.threshold,
            "is_trained": self.is_trained,
            "production_eligible": production_eligible,
            "metadata": meta
        }
        with open(output_path, "wb") as f:
            pickle.dump(artifact, f)
        print(f"[StackingPipeline] Saved artifact (production_eligible={production_eligible}, dataset_type='{dataset_type}') to: {output_path}", flush=True)

    @classmethod
    def load_artifact(
        cls,
        path: Path = config.MODELS_DIR / "fusion" / "stacking_ensemble.pkl",
        require_production_eligible: bool = True
    ) -> Optional["StackingEnsemblePipeline"]:
        """
        Safely loads pipeline artifact with strict production provenance gating.
        Rejects synthetic or unverified artifacts when require_production_eligible=True.
        Raises IneligibleProductionArtifactError if artifact fails validation.
        """
        path = Path(path)
        if not path.exists():
            return None

        with open(path, "rb") as f:
            artifact = pickle.load(f)

        is_eligible = bool(artifact.get("production_eligible", False))
        meta = artifact.get("metadata", {})
        dataset_type = meta.get("dataset_type", "")

        # Strict Production Gate: Refuse synthetic artifacts
        if require_production_eligible:
            if not is_eligible or dataset_type == "synthetic_simulation" or "synthetic" in str(path).lower():
                msg = (
                    f"Artifact at {path} is not eligible for production inference "
                    f"(production_eligible={is_eligible}, dataset_type='{dataset_type}'). "
                    "Production inference requires an independently verified empirical multimodal dataset."
                )
                print(f"[StackingPipeline] REFUSED: {msg}", flush=True)
                raise IneligibleProductionArtifactError(msg)

        pipeline = cls(
            meta_learner=artifact.get("meta_learner"),
            feature_names=artifact.get("feature_names"),
            threshold=artifact.get("threshold", config.DEFAULT_DECISION_THRESHOLD)
        )
        pipeline.baseline = artifact.get("baseline", WeightedLateFusionBaseline())
        pipeline.is_trained = artifact.get("is_trained", False)
        pipeline.production_eligible = is_eligible
        pipeline.training_metadata = meta
        return pipeline
