"""
Stacking Ensemble Training and Academic Benchmark Workflows.
Explicitly separates:
A. Synthetic Simulation Demonstration (run_synthetic_simulation_demo):
   - Demonstrates OOF cross-validation and meta-learner mathematical mechanics.
   - Saves artifacts ONLY in models/fusion/synthetic_demo/ and reports/synthetic_simulation/.
   - Marked as synthetic_simulation (production_eligible = False).
B. Production Meta-Model Training (train_production_stacking_model):
   - Accepts strictly an authentic empirical multimodal dataset.
   - Enforces synchronized branch observations, verified threat outcome labels, and non-circularity.
   - Rejects synthetic data and simulation tokens.
   - If no valid empirical dataset exists, stops with a descriptive error and leaves meta-model untrained.
"""

import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

# Ensure project root is in sys.path
_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import config
from src.fusion.meta_learner import (
    LogisticRegressionMetaLearner,
    RandomForestMetaLearner,
    XGBoostMetaLearner,
    WeightedLateFusionBaseline,
    HAS_XGBOOST
)
from src.fusion.oof_generator import OOFGenerator
from src.fusion.stacking_pipeline import (
    StackingEnsemblePipeline,
    DataCircularityError,
    MissingGroundTruthLabelError,
    SyntheticDatasetRejectedError
)
from src.fusion.evaluation import (
    evaluate_predictions,
    format_comparison_table,
    save_evaluation_report
)

def run_synthetic_simulation_demo(
    n_samples: int = 4000,
    test_ratio: float = 0.20,
    random_state: int = 42,
    artifact_output_dir: Path = config.MODELS_DIR / "fusion" / "synthetic_demo",
    reports_output_dir: Path = Path("reports/synthetic_simulation")
) -> Dict[str, Any]:
    """
    Workflow A: Synthetic Simulation Demonstration.
    Demonstrates OOF cross-validation and Level 1 Meta-Learner mechanics on an in-memory
    synthetic simulation. Saves all artifacts exclusively to isolated simulation directories.
    Marked strictly as: production_eligible = False, dataset_type = 'synthetic_simulation'.
    """
    np.random.seed(random_state)
    artifact_output_dir.mkdir(parents=True, exist_ok=True)
    reports_output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*70)
    print("DEMO SIMULATION: MULTIMODAL STACKING ENSEMBLE PIPELINE MECHANICS")
    print("NOTICE: This simulation uses in-memory synthetic variables for demonstration.")
    print("        Outputs are marked as synthetic_simulation (production_eligible=False).")
    print("="*70)

    # 1. Generate in-memory synthetic variables
    p_phys = np.random.beta(a=1.2, b=2.5, size=n_samples)
    p_spat = np.random.beta(a=1.5, b=2.2, size=n_samples)
    p_env = np.random.beta(a=1.0, b=2.8, size=n_samples)

    interaction_hazard = (
        2.5 * (p_phys * p_spat)
        + 1.8 * (p_spat * p_env)
        + 1.2 * p_phys
        + np.random.normal(0, 0.15, size=n_samples)
    )
    threat_prob_true = 1.0 / (1.0 + np.exp(-(interaction_hazard - 2.0)))
    threat_label = (threat_prob_true >= 0.50).astype(int)

    subjects = [f"Subj_{i:02d}" for i in range(1, 21)]
    subject_assignments = np.random.choice(subjects, size=n_samples)

    df_all = pd.DataFrame({
        "sample_id": [f"SAMP_{i:05d}" for i in range(n_samples)],
        "subject_id": subject_assignments,
        "physiological_stress_score": np.round(p_phys, 4),
        "location_risk_score": np.round(p_spat, 4),
        "environment_risk_score": np.round(p_env, 4),
        "ground_truth_threat_label": threat_label
    })

    # Subject-level train/test partition
    unique_subjs = np.unique(subject_assignments)
    n_test_subjs = int(len(unique_subjs) * test_ratio)
    test_subjs = set(unique_subjs[:n_test_subjs])
    train_subjs = set(unique_subjs[n_test_subjs:])

    train_mask = df_all["subject_id"].isin(train_subjs)
    test_mask = df_all["subject_id"].isin(test_subjs)

    df_train = df_all[train_mask].reset_index(drop=True)
    df_test = df_all[test_mask].reset_index(drop=True)

    print(f"Simulation Split: Train={len(df_train)} samples ({len(train_subjs)} subjects), Test={len(df_test)} samples ({len(test_subjs)} subjects)")

    feature_cols = config.STACKING_META_FEATURES
    X_train_meta = df_train[feature_cols].values
    y_train = df_train["ground_truth_threat_label"].values
    groups_train = df_train["subject_id"].values

    X_test_meta = df_test[feature_cols].values
    y_test = df_test["ground_truth_threat_label"].values

    # Out-of-fold matrix generation
    oof_gen = OOFGenerator(n_splits=5, random_state=random_state)
    cv, cv_name = oof_gen.select_cv_strategy(X_train_meta, y_train, groups_train)

    oof_lr = np.zeros(len(y_train))
    oof_rf = np.zeros(len(y_train))
    oof_xgb = np.zeros(len(y_train))

    splits_iter = cv.split(X_train_meta, y_train, groups_train)
    for fold, (tr_idx, va_idx) in enumerate(splits_iter):
        X_tr, y_tr = X_train_meta[tr_idx], y_train[tr_idx]
        X_va, y_va = X_train_meta[va_idx], y_train[va_idx]

        m_lr = LogisticRegressionMetaLearner(feature_names=feature_cols, random_state=random_state).fit(X_tr, y_tr)
        m_rf = RandomForestMetaLearner(feature_names=feature_cols, random_state=random_state).fit(X_tr, y_tr)
        oof_lr[va_idx] = m_lr.predict_proba(X_va)[:, 1]
        oof_rf[va_idx] = m_rf.predict_proba(X_va)[:, 1]
        if HAS_XGBOOST:
            m_xgb = XGBoostMetaLearner(feature_names=feature_cols, random_state=random_state).fit(X_tr, y_tr)
            oof_xgb[va_idx] = m_xgb.predict_proba(X_va)[:, 1]

    final_lr = LogisticRegressionMetaLearner(feature_names=feature_cols, random_state=random_state).fit(X_train_meta, y_train)
    final_rf = RandomForestMetaLearner(feature_names=feature_cols, random_state=random_state).fit(X_train_meta, y_train)
    if HAS_XGBOOST:
        final_xgb = XGBoostMetaLearner(feature_names=feature_cols, random_state=random_state).fit(X_train_meta, y_train)
    baseline = WeightedLateFusionBaseline()

    results = {}
    results["1. Physiological Only (Simulated)"] = evaluate_predictions(y_test, X_test_meta[:, 0], model_name="Physiological_Simulation")
    results["2. Spatial Only (Simulated)"] = evaluate_predictions(y_test, X_test_meta[:, 1], model_name="Spatial_Simulation")
    results["3. Environmental Only (Simulated)"] = evaluate_predictions(y_test, X_test_meta[:, 2], model_name="Environmental_Simulation")
    results["4. Weighted Late Fusion Baseline"] = evaluate_predictions(y_test, baseline.predict_proba(X_test_meta)[:, 1], model_name="Weighted_Late_Fusion_Baseline")
    results["5. Stacking (Logistic Regression Demo)"] = evaluate_predictions(y_test, final_lr.predict_proba(X_test_meta)[:, 1], model_name="Stacking_LogisticRegression_Demo")
    results["6. Stacking (Random Forest Demo)"] = evaluate_predictions(y_test, final_rf.predict_proba(X_test_meta)[:, 1], model_name="Stacking_RandomForest_Demo")
    if HAS_XGBOOST:
        results["7. Stacking (XGBoost Demo)"] = evaluate_predictions(y_test, final_xgb.predict_proba(X_test_meta)[:, 1], model_name="Stacking_XGBoost_Demo")

    df_cmp = format_comparison_table(results)
    print("\nSIMULATION DEMO METRICS (Not representative of real threat prediction):")
    print(df_cmp.to_string(index=False))

    # Save reports in isolated reports/synthetic_simulation/
    save_evaluation_report(results, output_dir=reports_output_dir, filename_prefix="stacking_benchmark_results_synthetic_demo")

    # Save isolated synthetic demo artifact
    demo_pipeline = StackingEnsemblePipeline(meta_learner=final_lr, feature_names=feature_cols)
    demo_pipeline.is_trained = True
    demo_pipeline.training_metadata = {
        "pipeline_version": StackingEnsemblePipeline.VERSION,
        "trained_at": pd.Timestamp.now().isoformat(),
        "cv_strategy": cv_name,
        "n_samples_train": len(df_train),
        "n_samples_test": len(df_test),
        "test_metrics": results["5. Stacking (Logistic Regression Demo)"],
        "all_comparison_results": results,
        "meta_learner_explain": final_lr.explain(),
        "dataset_type": "synthetic_simulation",
        "production_eligible": False,
        "dataset_source": "In-memory NumPy Beta distribution simulation",
        "target_definition": "Thresholded synthetic hazard equation",
        "target_provenance": "Synthetic simulation (non-empirical)"
    }
    demo_pipeline.save_artifact(
        output_path=artifact_output_dir / "stacking_ensemble_demo.pkl",
        dataset_source="In-memory NumPy Beta simulation",
        dataset_type="synthetic_simulation",
        target_definition="Thresholded synthetic hazard equation",
        target_provenance="Synthetic simulation",
        production_eligible=False
    )
    print(f"[Simulation] Artifact saved to demo directory: {artifact_output_dir / 'stacking_ensemble_demo.pkl'}")
    return results

# Backward compatibility alias
run_academic_benchmark_experiment = run_synthetic_simulation_demo

def train_production_stacking_model(
    empirical_dataset_path: Optional[Path] = None,
    output_model_path: Path = config.MODELS_DIR / "fusion" / "stacking_ensemble.pkl",
    target_col: str = "ground_truth_threat_label",
    group_col: Optional[str] = "subject_id"
) -> Dict[str, Any]:
    """
    Workflow B: Production Meta-Model Training.
    Accepts ONLY a genuine empirical multimodal dataset with verified ground-truth threat events.
    Rejects synthetic data, simulation tokens, and circular weighted formulas.
    If no empirical dataset is provided, stops with an error and leaves the production model untrained.
    """
    if empirical_dataset_path is None or not Path(empirical_dataset_path).exists():
        raise FileNotFoundError(
            "Production Training Stopped: No verified empirical multimodal threat dataset found.\n"
            "The production Stacking Meta-Learner cannot be trained without synchronized biometric, "
            "geospatial, and environmental telemetry accompanied by independently verified threat outcome labels.\n"
            "Meta-Learner status remains: UNTRAINED — VALID MULTIMODAL THREAT DATASET REQUIRED."
        )

    print(f"\n[ProductionTraining] Loading empirical dataset from: {empirical_dataset_path}...", flush=True)
    df = pd.read_csv(empirical_dataset_path)

    # Strictly enforce empirical validation (rejects synthetic tokens or circular labels)
    pipeline = StackingEnsemblePipeline()
    pipeline.validate_multimodal_dataset(
        df,
        target_col=target_col,
        enforce_production_empirical=True
    )

    print(f"[ProductionTraining] Empirical dataset validated. Training production meta-model...", flush=True)
    metadata = pipeline.train_with_oof(
        df_train=df,
        target_col=target_col,
        group_col=group_col,
        enforce_production_empirical=True
    )

    pipeline.save_artifact(
        output_path=output_model_path,
        dataset_source=str(empirical_dataset_path),
        dataset_type="empirical_multimodal",
        target_definition="Verified physical security / threat incident",
        target_provenance="Empirical incident logs",
        production_eligible=True
    )
    print(f"[ProductionTraining] Successfully saved production artifact to: {output_model_path}", flush=True)
    return metadata

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--train-production":
        dataset_arg = sys.argv[2] if len(sys.argv) > 2 else None
        try:
            train_production_stacking_model(dataset_arg)
        except Exception as e:
            print(f"[ProductionTraining Error] {e}")
            sys.exit(1)
    else:
        # Default runs the simulation demo
        run_synthetic_simulation_demo()
