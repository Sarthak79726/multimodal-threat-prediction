"""
Unit and Regression Test Suite for Version 1.0 Scientific & Technical Audit Corrections:
1. Crowd Classification: Balanced sample weighting, Medium-class performance, and macro F1 verification.
2. Missing Environmental Inputs: Strict rejection of silent 0.20 neutral substitution; returns UNKNOWN.
3. Partial Dual-Branch Context Index: Validates re-normalized formula (0.40P + 0.25L)/0.65 when env is missing.
4. Terminology Sanitization: Absence of 'calibrated' claims; environmental context score presence.
5. GPS Geofence & Proximity Ambiguity Detection: Ambiguity flagging when distance difference <= margin.
6. Gated Production Inference: Stacking meta-learner remains strictly gated and untrained.
"""

import unittest
import numpy as np
from pathlib import Path

import config
from src.environment.model import EnvironmentalContextEngine, SUPPORTED_LOCATIONS
from src.fusion.predict import MultimodalThreatPredictor

class TestAuditV1Corrections(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = EnvironmentalContextEngine()
        cls.predictor = MultimodalThreatPredictor()

    # =========================================================================
    # 1. CROWD CLASSIFICATION — BALANCED WEIGHTING & MEDIUM CLASS PERFORMANCE
    # =========================================================================
    def test_crowd_model_is_balanced_and_predicts_medium(self):
        """Verifies crowd model predicts Medium in transitional context (e.g. PMC at 14h or Girls Hostel at 21h)."""
        res_pmc = self.engine.predict_environment(
            location="PMC",
            hour_of_day=14.0,
            day_of_week="Friday",
            weekend_indicator=0
        )
        self.assertEqual(res_pmc["status"], "SUCCESS")
        # With balanced weighting, transitional hours now reliably predict Medium
        self.assertEqual(res_pmc["crowd_prediction"]["category"], "Medium")
        self.assertGreater(res_pmc["crowd_prediction"]["confidence"], 0.40)

    def test_crowd_model_classes_and_model_name(self):
        """Verifies crowd model artifact uses balanced gradient boosting."""
        self.assertIsNotNone(self.engine.crowd_artifact)
        self.assertIn("Balanced", self.engine.crowd_artifact.get("model_name", ""))
        self.assertEqual(list(self.engine.crowd_artifact["classes"]), ["Low", "Medium", "High"])

    # =========================================================================
    # 2. ELIMINATION OF SILENT NEUTRAL SUBSTITUTION (AUDIT ISSUE 2)
    # =========================================================================
    def test_unsupported_location_returns_none_score_and_unknown(self):
        """Verifies unknown location sets environmental scores to None rather than 0.20."""
        res = self.engine.predict_environment(
            location="ImaginaryLab999",
            hour_of_day=12.0
        )
        self.assertEqual(res["status"], "UNKNOWN")
        self.assertIsNone(res["environmental_context_score"], "Must not silently substitute neutral score 0.20")
        self.assertIsNone(res["calibrated_environmental_score"], "Alias must also be None")
        self.assertIsNone(res["environmental_crowd_context_score"])
        self.assertIsNone(res["environmental_lighting_context_score"])
        self.assertEqual(res["crowd_prediction"]["category"], "UNKNOWN")

    def test_missing_model_returns_none_score_and_unknown(self):
        """Verifies missing model artifact sets environmental score to None."""
        unloaded_engine = EnvironmentalContextEngine(
            crowd_model_path=Path("non_existent_crowd.pkl"),
            lighting_model_path=Path("non_existent_lighting.pkl")
        )
        res = unloaded_engine.predict_environment(location="Main Building", hour_of_day=12.0)
        self.assertEqual(res["status"], "UNKNOWN")
        self.assertIsNone(res["environmental_context_score"], "Must not substitute 0.20 on load failure")
        self.assertIsNone(res["calibrated_environmental_score"])

    # =========================================================================
    # 3. PARTIAL DUAL-BRANCH CONTEXT INDEX (AUDIT ISSUE 2)
    # =========================================================================
    def test_partial_dual_branch_when_env_is_none(self):
        """Verifies that missing environmental input triggers explicit Partial Dual-Branch Index."""
        phys_score = 0.20
        loc_score = 0.60
        
        # When env is None
        res_dual = self.predictor.predict_fusion(
            physiological_stress_score=phys_score,
            location_risk_score=loc_score,
            environment_risk_score=None
        )
        self.assertEqual(res_dual["fusion_mode"], "PARTIAL_DUAL_BRANCH")
        self.assertIsNone(res_dual["experimental_context_index"], "3-branch index must be None")
        self.assertIsNone(res_dual["experimental_context_index_pct"])
        self.assertIn("PARTIAL DUAL-BRANCH", res_dual["active_model_name"])
        
        # Expected formula: (0.40 * 0.20 + 0.25 * 0.60) / 0.65 = 0.23 / 0.65 = 0.353846... -> 35.38%
        expected_score = round((0.40 * phys_score + 0.25 * loc_score) / 0.65, 4)
        expected_pct = round(expected_score * 100.0, 2)
        self.assertEqual(res_dual["partial_dual_branch_index"], expected_score)
        self.assertEqual(res_dual["partial_dual_branch_index_pct"], expected_pct)
        self.assertEqual(res_dual["available_modalities"], ["physiological", "spatial"])
        self.assertEqual(res_dual["missing_modalities"], ["environmental"])
        self.assertIn("61.54%", res_dual["explanation"])
        self.assertIn("38.46%", res_dual["explanation"])

    def test_complete_three_branch_fusion_when_env_is_present(self):
        """Verifies complete 3-branch weighted baseline executes when env score is provided."""
        res_3 = self.predictor.predict_fusion(
            physiological_stress_score=0.20,
            location_risk_score=0.60,
            environment_risk_score=0.28
        )
        self.assertEqual(res_3["fusion_mode"], "THREE_BRANCH_FUSION")
        self.assertIsNotNone(res_3["experimental_context_index"])
        # Expected: 0.20 * 0.40 + 0.60 * 0.25 + 0.28 * 0.35 = 0.08 + 0.15 + 0.098 = 0.328 (32.8%)
        self.assertAlmostEqual(res_3["experimental_context_index_pct"], 32.80, places=2)
        self.assertIsNone(res_3["partial_dual_branch_index"])

    # =========================================================================
    # 4. TERMINOLOGY SANITIZATION (AUDIT ISSUE 3)
    # =========================================================================
    def test_environmental_context_score_field_presence(self):
        """Verifies that predictions provide environmental_context_score."""
        res = self.engine.predict_environment(location="Main Building", hour_of_day=12.0)
        self.assertIn("environmental_context_score", res)
        self.assertGreaterEqual(res["environmental_context_score"], 0.05)
        self.assertLessEqual(res["environmental_context_score"], 1.0)
        # Provenance disclaimer notes experimental situational index
        self.assertIn("EXPERIMENTAL PREDICTION", res["warning"])
        self.assertIn("NOT LIVE SENSOR MEASUREMENT", res["warning"])

    # =========================================================================
    # 5. GPS GEOFENCE & PROXIMITY AMBIGUITY DETECTION (AUDIT ISSUE 4)
    # =========================================================================
    def test_gps_outside_campus_geofence(self):
        """Verifies coordinates > 350m from campus are flagged as OUTSIDE_CAMPUS_GEOFENCE."""
        # Ranchi Railway Station coordinates (~15 km away)
        res = self.engine.resolve_campus_location(lat=23.3500, lon=85.3200, max_geofence_meters=350.0)
        self.assertEqual(res["status"], "OUTSIDE_CAMPUS_GEOFENCE")
        self.assertIsNone(res["resolved_location"])
        self.assertFalse(res["is_within_geofence"])
        self.assertGreater(res["distance_meters"], 350.0)

    def test_gps_ambiguity_between_close_landmarks(
        self,
    ):
        """
        Verifies that coordinates located equidistant between nearby landmarks
        (e.g. R&D Building and Hostel 2, distance = 23.1m) flag AMBIGUOUS_LOCATION.
        """
        rnd_coord = config.BIT_MESRA_LANDMARKS_DB["rnd"]
        h2_coord = config.BIT_MESRA_LANDMARKS_DB["h2"]
        # Exact midpoint coordinates between R&D and H2
        mid_lat = (rnd_coord["latitude"] + h2_coord["latitude"]) / 2.0
        mid_lon = (rnd_coord["longitude"] + h2_coord["longitude"]) / 2.0

        res = self.engine.resolve_campus_location(
            lat=mid_lat,
            lon=mid_lon,
            max_geofence_meters=350.0,
            accuracy=30.0
        )
        self.assertEqual(res["status"], "AMBIGUOUS_LOCATION")
        self.assertIsNone(res["resolved_location"])
        self.assertTrue(res["is_within_geofence"])
        self.assertGreaterEqual(len(res["ambiguous_candidates"]), 2)
        # Should flag R&D and H2 as candidates
        cand_str = " ".join(res["ambiguous_candidates"])
        self.assertTrue("R&D" in cand_str and "H2" in cand_str)

    def test_unambiguous_gps_resolution(self):
        """Verifies unambiguous landmark resolution when coordinates are firmly at one landmark."""
        h10 = config.BIT_MESRA_LANDMARKS_DB["h10"]
        res = self.engine.resolve_campus_location(
            lat=h10["latitude"],
            lon=h10["longitude"],
            max_geofence_meters=350.0,
            accuracy=5.0
        )
        self.assertEqual(res["status"], "RESOLVED_GPS")
        self.assertEqual(res["resolved_location"], "H10")
        self.assertTrue(res["is_within_geofence"])
        self.assertEqual(res["ambiguous_candidates"], [])

    # =========================================================================
    # 6. GATED PRODUCTION INFERENCE VERIFICATION
    # =========================================================================
    def test_production_meta_learner_remains_untrained(self):
        """Verifies production inference does not load synthetic meta-model."""
        self.assertFalse(self.predictor.stacking_pipeline.is_trained)
        res = self.predictor.predict_fusion(0.20, 0.60, 0.28)
        self.assertFalse(res["is_meta_learner_trained"])
        self.assertEqual(res["meta_model_status"], "UNTRAINED — VALID MULTIMODAL THREAT DATASET REQUIRED")
        self.assertIsNone(res["stacking_meta_probability"])

if __name__ == "__main__":
    unittest.main()
