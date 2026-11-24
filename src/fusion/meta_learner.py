"""
Meta-Learner Implementations for Stacking Ensemble Fusion.
Implements Level 1 Meta-Learners:
- LogisticRegressionMetaLearner (Primary Stacker with full mathematical explainability)
- RandomForestMetaLearner (Non-linear baseline stacker)
- XGBoostMetaLearner (Gradient-boosted baseline stacker)
- WeightedLateFusionBaseline (Explicit un-trained reference benchmark)
"""

import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

# Ensure project root is in sys.path
_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import config

class BaseMetaLearner:
    """Abstract interface for Level 1 Stacking Meta-Learners."""
    def __init__(self, feature_names: Optional[List[str]] = None):
        self.feature_names = feature_names or config.STACKING_META_FEATURES
        self.is_trained_ml: bool = False
        self.model_name: str = "BaseMetaLearner"

    def fit(self, X_meta: np.ndarray, y: np.ndarray) -> "BaseMetaLearner":
        raise NotImplementedError

    def predict_proba(self, X_meta: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def predict(self, X_meta: np.ndarray, threshold: float = 0.50) -> np.ndarray:
        probs = self.predict_proba(X_meta)[:, 1]
        return (probs >= threshold).astype(int)

    def explain(self, sample_vec: Optional[np.ndarray] = None) -> Dict[str, Any]:
        raise NotImplementedError

class LogisticRegressionMetaLearner(BaseMetaLearner):
    """
    Primary Level 1 Stacking Meta-Learner based on Logistic Regression.
    
    Mathematical Formulation:
        X_meta = [P_physiological, P_spatial, P_environmental]
        z = beta_0 + sum_{i=1}^3 beta_i * P_i
        P_final = 1 / (1 + exp(-z))
        
    Learns beta coefficients directly from out-of-fold meta-features.
    Provides transparent mathematical explainability (direction, magnitude, odds ratios, contribution decomposition).
    """
    def __init__(
        self,
        feature_names: Optional[List[str]] = None,
        C: float = 1.0,
        random_state: int = 42
    ):
        super().__init__(feature_names)
        self.model_name = "LogisticRegression_MetaLearner"
        self.C = C
        self.random_state = random_state
        self.classifier = LogisticRegression(
            C=self.C,
            solver="lbfgs",
            random_state=self.random_state,
            max_iter=1000
        )
        self.intercept_: float = 0.0
        self.coef_: np.ndarray = np.array([])
        self.is_trained_ml = False

    def fit(self, X_meta: np.ndarray, y: np.ndarray) -> "LogisticRegressionMetaLearner":
        X_arr = np.asarray(X_meta, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.int32)
        
        if X_arr.ndim != 2 or X_arr.shape[1] != len(self.feature_names):
            raise ValueError(f"X_meta shape {X_arr.shape} does not match feature dimension {len(self.feature_names)}")
        if len(np.unique(y_arr)) < 2:
            raise ValueError("Cannot train meta-learner with only a single target class present.")

        self.classifier.fit(X_arr, y_arr)
        self.intercept_ = float(self.classifier.intercept_[0])
        self.coef_ = self.classifier.coef_[0].copy()
        self.is_trained_ml = True
        return self

    def predict_proba(self, X_meta: np.ndarray) -> np.ndarray:
        if not self.is_trained_ml:
            raise RuntimeError("Meta-learner is not fitted yet. Call fit() first.")
        X_arr = np.asarray(X_meta, dtype=np.float64)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)
        return self.classifier.predict_proba(X_arr)

    def explain(self, sample_vec: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Generates mathematical explainability report:
        - beta_0 (intercept / baseline log-odds)
        - beta_i coefficients (direction and magnitude)
        - Odds ratios exp(beta_i)
        - Local linear contribution z_i = beta_i * P_i (if sample_vec provided)
        """
        if not self.is_trained_ml:
            return {
                "status": "NOT_TRAINED",
                "message": "Meta-learner has not been trained on valid out-of-fold data."
            }

        coef_dict = dict(zip(self.feature_names, [float(c) for c in self.coef_]))
        odds_ratios = {k: float(np.exp(v)) for k, v in coef_dict.items()}

        interpretation = []
        for name, coef in coef_dict.items():
            direction = "elevates" if coef > 0 else "reduces"
            interpretation.append(
                f"{name} (beta={coef:+.4f}, OR={odds_ratios[name]:.3f}) {direction} final threat log-odds."
            )

        report = {
            "model_type": "Logistic Regression Stacking Meta-Learner (Level 1)",
            "intercept_beta0": self.intercept_,
            "coefficients": coef_dict,
            "odds_ratios": odds_ratios,
            "interpretation": " | ".join(interpretation),
            "scientific_note": "Logistic regression coefficients quantify associative log-odds weighting on the meta-feature distribution, not proof of causal importance."
        }

        if sample_vec is not None:
            s_arr = np.asarray(sample_vec, dtype=np.float64).ravel()
            if len(s_arr) == len(self.feature_names):
                linear_terms = {name: float(self.coef_[i] * s_arr[i]) for i, name in enumerate(self.feature_names)}
                z = self.intercept_ + sum(linear_terms.values())
                p = 1.0 / (1.0 + np.exp(-z))
                total_abs_contrib = sum(abs(v) for v in linear_terms.values()) + 1e-9
                rel_contrib = {k: float(abs(v) / total_abs_contrib * 100.0) for k, v in linear_terms.items()}

                report["sample_evaluation"] = {
                    "input_meta_vector": dict(zip(self.feature_names, [float(x) for x in s_arr])),
                    "linear_terms_z_i": linear_terms,
                    "latent_logit_z": float(z),
                    "final_probability_P": float(p),
                    "relative_contribution_pct": rel_contrib
                }

        return report

class RandomForestMetaLearner(BaseMetaLearner):
    """Random Forest Meta-Learner benchmark for non-linear stacking comparisons."""
    def __init__(
        self,
        feature_names: Optional[List[str]] = None,
        n_estimators: int = 100,
        max_depth: int = 4,
        random_state: int = 42
    ):
        super().__init__(feature_names)
        self.model_name = "RandomForest_MetaLearner"
        self.classifier = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state
        )
        self.feature_importances_: Dict[str, float] = {}

    def fit(self, X_meta: np.ndarray, y: np.ndarray) -> "RandomForestMetaLearner":
        X_arr = np.asarray(X_meta, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.int32)
        self.classifier.fit(X_arr, y_arr)
        self.feature_importances_ = dict(zip(self.feature_names, [float(fi) for fi in self.classifier.feature_importances_]))
        self.is_trained_ml = True
        return self

    def predict_proba(self, X_meta: np.ndarray) -> np.ndarray:
        if not self.is_trained_ml:
            raise RuntimeError("Meta-learner is not fitted.")
        X_arr = np.asarray(X_meta, dtype=np.float64)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)
        return self.classifier.predict_proba(X_arr)

    def explain(self, sample_vec: Optional[np.ndarray] = None) -> Dict[str, Any]:
        return {
            "model_type": "Random Forest Stacking Meta-Learner (Non-linear Benchmark)",
            "feature_importances": self.feature_importances_
        }

class XGBoostMetaLearner(BaseMetaLearner):
    """XGBoost Meta-Learner benchmark for gradient boosting stacking comparisons."""
    def __init__(
        self,
        feature_names: Optional[List[str]] = None,
        n_estimators: int = 100,
        max_depth: int = 3,
        learning_rate: float = 0.05,
        random_state: int = 42
    ):
        super().__init__(feature_names)
        self.model_name = "XGBoost_MetaLearner"
        if not HAS_XGBOOST:
            self.classifier = None
        else:
            self.classifier = XGBClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                eval_metric="logloss",
                random_state=random_state
            )
        self.feature_importances_: Dict[str, float] = {}

    def fit(self, X_meta: np.ndarray, y: np.ndarray) -> "XGBoostMetaLearner":
        if not HAS_XGBOOST:
            raise ImportError("XGBoost is not installed.")
        X_arr = np.asarray(X_meta, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.int32)
        self.classifier.fit(X_arr, y_arr)
        self.feature_importances_ = dict(zip(self.feature_names, [float(fi) for fi in self.classifier.feature_importances_]))
        self.is_trained_ml = True
        return self

    def predict_proba(self, X_meta: np.ndarray) -> np.ndarray:
        if not self.is_trained_ml:
            raise RuntimeError("Meta-learner is not fitted.")
        X_arr = np.asarray(X_meta, dtype=np.float64)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)
        return self.classifier.predict_proba(X_arr)

    def explain(self, sample_vec: Optional[np.ndarray] = None) -> Dict[str, Any]:
        return {
            "model_type": "XGBoost Stacking Meta-Learner (Gradient Boosting Benchmark)",
            "feature_importances": self.feature_importances_
        }

class WeightedLateFusionBaseline:
    """
    Explicitly identified Un-Trained Weighted Late Fusion Reference Baseline.
    Applies fixed heuristic weights:
        S = w_physio * P_physio + w_loc * P_loc + w_env * P_env
    Default weights: 0.40 Physiological, 0.25 Location, 0.35 Environment.
    
    This is an experimental baseline benchmark, NOT a trained machine learning model.
    """
    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        threshold: float = 0.50
    ):
        self.weights_dict = weights or config.DEFAULT_BASELINE_WEIGHTS
        self.threshold = threshold
        self.is_trained_ml = False
        self.model_name = "Weighted Late Fusion (Non-Trained Baseline)"
        
        # Normalize weights
        total = sum(self.weights_dict.values())
        self.w_physio = self.weights_dict.get("physiological", 0.40) / total
        self.w_loc = self.weights_dict.get("location", 0.25) / total
        self.w_env = self.weights_dict.get("environment", 0.35) / total

    def predict_score(
        self,
        p_physio: float,
        p_loc: float,
        p_env: float
    ) -> float:
        score = (
            self.w_physio * float(np.clip(p_physio, 0.0, 1.0)) +
            self.w_loc * float(np.clip(p_loc, 0.0, 1.0)) +
            self.w_env * float(np.clip(p_env, 0.0, 1.0))
        )
        return float(min(1.0, max(0.0, score)))

    def predict_proba(self, X_meta: np.ndarray) -> np.ndarray:
        """Emulates predict_proba format for comparative evaluation."""
        X_arr = np.asarray(X_meta, dtype=np.float64)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)
        scores = (
            self.w_physio * X_arr[:, 0] +
            self.w_loc * X_arr[:, 1] +
            self.w_env * X_arr[:, 2]
        )
        scores = np.clip(scores, 0.0, 1.0)
        return np.column_stack([1.0 - scores, scores])

    def predict(self, X_meta: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X_meta)[:, 1]
        return (probs >= self.threshold).astype(int)

    def explain(self, sample_vec: Optional[np.ndarray] = None) -> Dict[str, Any]:
        return {
            "model_type": "Weighted Late Fusion (Non-Trained Baseline)",
            "is_trained_ml": False,
            "weights": {
                "physiological": self.w_physio,
                "location": self.w_loc,
                "environment": self.w_env
            },
            "threshold": self.threshold,
            "scientific_note": "This baseline employs fixed, non-learned empirical weights for benchmarking purposes only."
        }
