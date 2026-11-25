"""
Integration Tests for Unified Multimodal Threat Predictor.
Tests:
- End-to-end inference across physiological, spatial, environmental branches.
- Active decision selection (Stacking Meta-Learner vs Late Fusion Baseline).
- Provenance tracking (verifying is_trained_ml flags for each branch).
- Backward compatibility of return dictionary fields.
"""

import unittest
import numpy as np
import config
from src.fusion.predict import MultimodalThreatPredictor

class TestPredictorIntegration(unittest.TestCase):
    def setUp(self):
        self.predictor = MultimodalThreatPredictor()

    def test_full_fusion_inference_schema(self):
        """Verifies that predict_fusion returns guaranteed schema with both Stacking and Baseline values."""
        res = self.predictor.predict_fusion(
            physiological_stress_score=0.30,
            location_risk_score=0.40,
            environment_risk_score=0.50,
            threshold=0.50
        )

        # Essential keys
        self.assertIn("physiological_stress_score", res)
        self.assertIn("location_risk_score", res)
        self.assertIn("environment_risk_score", res)
        self.assertIn("threat_probability", res)
        self.assertIn("threat_score_pct", res)
        self.assertIn("threat_status", res)
        self.assertIn("active_model_name", res)
        self.assertIn("is_meta_learner_trained", res)
        self.assertIn("modalities_metadata", res)
        self.assertIn("baseline_threat_probability", res)
        self.assertIn("explanation", res)

        # Verify probability bounds and context status indicators
        self.assertGreaterEqual(res["threat_probability"], 0.0)
        self.assertLessEqual(res["threat_probability"], 1.0)
        self.assertIn(res["threat_status"], ["STANDARD CONTEXT", "ELEVATED CONTEXT", "HIGH CONTEXT"])
        self.assertIn(res["context_status"], ["STANDARD CONTEXT", "ELEVATED CONTEXT", "HIGH CONTEXT"])
        self.assertIn(res["context_classification"], ["STANDARD CONTEXT", "ELEVATED CONTEXT", "HIGH CONTEXT"])
        self.assertFalse(res["is_meta_learner_trained"])
        self.assertEqual(res["meta_model_status"], "UNTRAINED — VALID MULTIMODAL THREAT DATASET REQUIRED")

    def test_branch_provenance_transparency(self):
        """Verifies that each branch transparently reports its machine learning vs rule-based nature."""
        res = self.predictor.predict_fusion(0.10, 0.20, 0.30)
        meta = res["modalities_metadata"]

        # Physiological should be trained ML (WESAD Random Forest)
        self.assertTrue(meta["physiological"]["is_trained_ml"])
        self.assertEqual(meta["physiological"]["branch"], "Physiological")

        # Spatial should be flagged as deterministic / rule-based by default
        self.assertFalse(meta["spatial"]["is_trained_ml"])
        self.assertEqual(meta["spatial"]["branch"], "Spatial")

        # Environmental is now an AI-based experimental predictor (trained on synthetic campus dataset)
        self.assertTrue(meta["environmental"]["is_trained_ml"])
        self.assertEqual(meta["environmental"]["branch"], "Environmental")
        self.assertEqual(meta["environmental"]["synthetic_provenance"], "SYNTHETIC_SIMULATION_RULE_GENERATED")

    def test_missing_modality_robustness(self):
        """Verifies predictor gracefully handles extreme inputs without throwing exceptions."""
        # Extreme negative / positive inputs
        res_neg = self.predictor.predict_fusion(-1.0, 2.5, float("inf"))
        self.assertGreaterEqual(res_neg["threat_probability"], 0.0)
        self.assertLessEqual(res_neg["threat_probability"], 1.0)

if __name__ == "__main__":
    unittest.main()

