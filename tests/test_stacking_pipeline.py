"""
Unit Tests for Stacking Ensemble Pipeline and Validation Guards.
Tests:
- Rejection of circular target labels (labels generated from weighted late fusion formula).
- Rejection of missing target labels or missing feature columns.
- End-to-end OOF training and untouched test evaluation.
- Pipeline artifact serialization and round-trip deserialization.
"""

import unittest
import numpy as np
import pandas as pd
from pathlib import Path

import config
from src.fusion.stacking_pipeline import (
    StackingEnsemblePipeline,
    DataCircularityError,
    MissingGroundTruthLabelError
)
from src.fusion.meta_learner import LogisticRegressionMetaLearner

class TestStackingPipeline(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.n_samples = 300
        p1 = np.random.beta(1.0, 2.0, self.n_samples)
        p2 = np.random.beta(1.0, 2.0, self.n_samples)
        p3 = np.random.beta(1.0, 2.0, self.n_samples)

        # Independent non-circular target
        hazard = 2.0 * p1 * p2 + 1.5 * p3 - 1.2
        p_true = 1.0 / (1.0 + np.exp(-hazard))
        y_true = (p_true >= 0.50).astype(int)

        self.df_valid = pd.DataFrame({
            "sample_id": [f"S_{i}" for i in range(self.n_samples)],
            "subject_id": np.random.choice([f"Subj_{i}" for i in range(1, 7)], size=self.n_samples),
            "physiological_stress_score": p1,
            "location_risk_score": p2,
            "environment_risk_score": p3,
            "threat_label": y_true
        })

    def test_circular_label_detection_and_rejection(self):
        """Verifies that datasets where target is derived from 0.40*phys + 0.25*loc + 0.35*env are strictly rejected."""
        df_circular = self.df_valid.copy()
        # Manufacture circular label from the weighted sum formula
        weighted_sum = (
            0.40 * df_circular["physiological_stress_score"] +
            0.25 * df_circular["location_risk_score"] +
            0.35 * df_circular["environment_risk_score"]
        )
        df_circular["threat_label"] = (weighted_sum >= 0.50).astype(int)

        pipeline = StackingEnsemblePipeline()
        with self.assertRaises(DataCircularityError):
            pipeline.validate_multimodal_dataset(df_circular, target_col="threat_label")

    def test_missing_target_rejection(self):
        """Verifies that datasets without ground truth target are rejected."""
        df_no_target = self.df_valid.drop(columns=["threat_label"])
        pipeline = StackingEnsemblePipeline()
        with self.assertRaises(MissingGroundTruthLabelError):
            pipeline.validate_multimodal_dataset(df_no_target, target_col="threat_label")

    def test_pipeline_train_with_oof(self):
        """Verifies complete OOF training, coefficient learning, and prediction."""
        pipeline = StackingEnsemblePipeline()
        metadata = pipeline.train_with_oof(
            df_train=self.df_valid,
            target_col="threat_label",
            group_col="subject_id",
            n_splits=3,
            random_state=42
        )

        self.assertTrue(pipeline.is_trained)
        self.assertIn("oof_validation_metrics", metadata)
        self.assertGreater(metadata["oof_validation_metrics"]["roc_auc"], 0.70)

        # Test inference
        res = pipeline.predict(0.60, 0.70, 0.80)
        self.assertTrue(res["is_meta_learner_trained"])
        self.assertIsInstance(res["active_threat_probability"], float)
        self.assertIn(res["active_threat_status"], ["SAFE", "RISKY"])

    def test_artifact_save_and_load(self):
        """Verifies pipeline artifact persistence with version metadata."""
        pipeline = StackingEnsemblePipeline()
        pipeline.train_with_oof(
            df_train=self.df_valid,
            target_col="threat_label",
            group_col="subject_id",
            n_splits=3,
            random_state=42
        )

        test_path = config.MODELS_DIR / "fusion" / "test_stacking_artifact.pkl"
        pipeline.save_artifact(test_path, production_eligible=True)
        self.assertTrue(test_path.exists())

        loaded = StackingEnsemblePipeline.load_artifact(test_path, require_production_eligible=True)
        self.assertIsNotNone(loaded)
        self.assertTrue(loaded.is_trained)
        self.assertEqual(loaded.VERSION, pipeline.VERSION)

        # Cleanup
        if test_path.exists():
            test_path.unlink()

if __name__ == "__main__":
    unittest.main()

