"""
Comprehensive Test Suite for Environmental Context AI Prediction & Fusion Resilience (Scenarios A through G).

Scenarios Covered:
- Scenario A: All 3 branches valid -> complete 3-branch weighted index (0.40P + 0.25L + 0.35E).
- Scenario B: Environmental model files missing -> clean diagnostic error, E=None, fallback to partial dual-branch.
- Scenario C: Invalid campus location -> clear validation error, E=None, diagnostic warning logged.
- Scenario D: Missing time input -> automatic Asia/Kolkata IST fallback succeeds with valid float score.
- Scenario E: Environmental prediction succeeds -> fusion receives valid float E and evaluates 3-branch index.
- Scenario F: Non-finite values (NaN, Inf, strings, out-of-range) -> rejected safely without crash.
- Scenario G: Streamlit AppTest rerun -> session state preserved, latest result rendered without spurious errors.
"""

import sys
import unittest
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

# Ensure project root is in sys.path
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import config
from src.environment.model import EnvironmentalContextEngine, SUPPORTED_LOCATIONS
from src.fusion.predict import MultimodalThreatPredictor
from streamlit.testing.v1 import AppTest


class TestEnvironmentalPipelineResilience(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.predictor = MultimodalThreatPredictor()

    def test_scenario_a_all_three_branches_valid(self):
        """Scenario A: All 3 branches valid -> complete 3-branch weighted index (0.40P + 0.25L + 0.35E)."""
        phys_score = 0.20
        loc_score = 0.60
        env_score = 0.28
        res = self.predictor.predict_fusion(
            physiological_stress_score=phys_score,
            location_risk_score=loc_score,
            environment_risk_score=env_score,
            threshold=0.50
        )
        self.assertEqual(res["fusion_mode"], "THREE_BRANCH_FUSION")
        self.assertIsNotNone(res["experimental_context_index"])
        expected_index = round(0.40 * phys_score + 0.25 * loc_score + 0.35 * env_score, 4)
        self.assertAlmostEqual(res["experimental_context_index"], expected_index, places=3)
        self.assertEqual(res["context_status"], "STANDARD CONTEXT")
        self.assertIn("environmental", res["available_modalities"])
        self.assertEqual(len(res["missing_modalities"]), 0)

    def test_scenario_b_missing_model_files(self):
        """Scenario B: Environmental model files missing -> clean diagnostic error, E=None, fallback to partial dual-branch."""
        engine_no_models = EnvironmentalContextEngine(
            crowd_model_path=Path("non_existent_crowd.pkl"),
            lighting_model_path=Path("non_existent_lighting.pkl")
        )
        self.assertFalse(engine_no_models.is_trained_ml)
        self.assertIn("Model artifacts not found", engine_no_models.load_error)

        pred_res = engine_no_models.predict_environment(location="Main Building", hour_of_day=14.0)
        self.assertEqual(pred_res["status"], "UNKNOWN")
        self.assertEqual(pred_res["error"], "MODEL_UNAVAILABLE")
        self.assertIsNone(pred_res["environmental_context_score"])
        self.assertIsNone(pred_res["calibrated_environmental_score"])
        self.assertNotIn("EXPERIMENTAL PREDICTION", pred_res["warning"])

        # Fusion receives None for environmental score
        fusion_res = self.predictor.predict_fusion(
            physiological_stress_score=0.40,
            location_risk_score=0.30,
            environment_risk_score=pred_res["environmental_context_score"]
        )
        self.assertEqual(fusion_res["fusion_mode"], "PARTIAL_DUAL_BRANCH")
        self.assertIsNone(fusion_res["experimental_context_index"])
        self.assertIsNotNone(fusion_res["partial_dual_branch_index"])
        expected_partial = round((0.40 * 0.40 + 0.25 * 0.30) / 0.65, 4)
        self.assertAlmostEqual(fusion_res["partial_dual_branch_index"], expected_partial, places=3)

    def test_scenario_c_invalid_location(self):
        """Scenario C: Invalid campus location -> clear validation error, E=None, diagnostic warning logged."""
        engine = EnvironmentalContextEngine()
        pred_res = engine.predict_environment(location="Non_Existent_Location_XYZ", hour_of_day=14.0)
        self.assertEqual(pred_res["status"], "UNKNOWN")
        self.assertEqual(pred_res["error"], "UNSUPPORTED_LOCATION")
        self.assertIsNone(pred_res["environmental_context_score"])
        self.assertIn("not a recognized BIT Mesra campus landmark", pred_res["message"])
        self.assertNotIn("EXPERIMENTAL PREDICTION", pred_res["warning"])

        # Modality passed to fusion defaults to partial dual-branch
        fusion_res = self.predictor.predict_fusion(
            physiological_stress_score=0.25,
            location_risk_score=0.20,
            environment_risk_score=pred_res["environmental_context_score"]
        )
        self.assertEqual(fusion_res["fusion_mode"], "PARTIAL_DUAL_BRANCH")
        self.assertIsNone(fusion_res["experimental_context_index"])

    def test_scenario_d_missing_time_input(self):
        """Scenario D: Missing time input -> automatic Asia/Kolkata IST fallback succeeds with valid float score."""
        engine = EnvironmentalContextEngine()
        # Omit hour_of_day, day_of_week, weekend_indicator completely
        pred_res = engine.predict_environment(location="Main Building")
        self.assertEqual(pred_res["status"], "SUCCESS")
        self.assertIsNotNone(pred_res["environmental_context_score"])
        self.assertTrue(0.05 <= pred_res["environmental_context_score"] <= 1.0)
        self.assertIn("IST", pred_res["time_display"])
        self.assertIn("disclaimer", pred_res)
        self.assertEqual(pred_res["crowd_prediction"]["status"], "ACTIVE_ML_PREDICTION")
        self.assertEqual(pred_res["lighting_prediction"]["status"], "ACTIVE_ML_PREDICTION")

    def test_scenario_e_environmental_prediction_succeeds_and_feeds_fusion(self):
        """Scenario E: Environmental prediction succeeds -> fusion receives valid float E and evaluates 3-branch index."""
        env_payload = self.predictor.predict_environment_context(
            location="Sports Complex",
            hour=15.0,
            day_of_week="Wednesday",
            is_weekend=0
        )
        self.assertEqual(env_payload["status"], "SUCCESS")
        env_score = env_payload["environmental_context_score"]
        self.assertIsNotNone(env_score)
        self.assertTrue(np.isfinite(env_score))

        fusion_res = self.predictor.predict_fusion(
            physiological_stress_score=0.18,
            location_risk_score=0.25,
            environment_risk_score=env_score
        )
        self.assertEqual(fusion_res["fusion_mode"], "THREE_BRANCH_FUSION")
        self.assertIsNotNone(fusion_res["experimental_context_index"])
        self.assertAlmostEqual(
            fusion_res["experimental_context_index"],
            round(0.40 * 0.18 + 0.25 * 0.25 + 0.35 * env_score, 4),
            places=3
        )

    def test_scenario_f_non_finite_values_rejected(self):
        """Scenario F: Non-finite values (NaN, Inf, out-of-range) -> rejected safely without crash."""
        # Test NaN environment score
        res_nan = self.predictor.predict_fusion(
            physiological_stress_score=0.30,
            location_risk_score=0.20,
            environment_risk_score=float("nan")
        )
        self.assertEqual(res_nan["fusion_mode"], "PARTIAL_DUAL_BRANCH")
        self.assertIsNone(res_nan["experimental_context_index"])

        # Test Inf environment score
        res_inf = self.predictor.predict_fusion(
            physiological_stress_score=0.30,
            location_risk_score=0.20,
            environment_risk_score=float("inf")
        )
        self.assertEqual(res_inf["fusion_mode"], "PARTIAL_DUAL_BRANCH")
        self.assertIsNone(res_inf["experimental_context_index"])

        # Test out-of-range negative environment score
        res_neg = self.predictor.predict_fusion(
            physiological_stress_score=0.30,
            location_risk_score=0.20,
            environment_risk_score=-0.50
        )
        self.assertEqual(res_neg["fusion_mode"], "PARTIAL_DUAL_BRANCH")

        # Test out-of-range > 1.0 environment score
        res_high = self.predictor.predict_fusion(
            physiological_stress_score=0.30,
            location_risk_score=0.20,
            environment_risk_score=1.50
        )
        self.assertEqual(res_high["fusion_mode"], "PARTIAL_DUAL_BRANCH")

    def test_scenario_g_streamlit_rerun_and_session_state(self):
        """Scenario G: Streamlit AppTest rerun -> session state preserved, latest result rendered properly."""
        app_path = str(Path(_project_root) / "app" / "main.py")
        at = AppTest.from_file(app_path)
        at.run(timeout=30)
        self.assertEqual(len(at.exception), 0, f"App startup encountered exceptions: {at.exception}")

        # Check that env_score is present, valid float, and not None
        env_score_initial = at.session_state["env_score"]
        self.assertIsNotNone(env_score_initial)
        self.assertTrue(0.0 <= env_score_initial <= 1.0)

        # Click Evaluate button to compute fusion
        btn = at.button(key="btn_main_threat")
        btn.click().run(timeout=30)
        self.assertEqual(len(at.exception), 0, f"Button execution failed: {at.exception}")

        threat = at.session_state["analyzed_threat"]
        self.assertIsNotNone(threat)
        self.assertEqual(threat["fusion_mode"], "THREE_BRANCH_FUSION")
        self.assertIsNotNone(threat["experimental_context_index"])

        # Perform a second rerun and verify session state is preserved
        at.run(timeout=30)
        self.assertEqual(len(at.exception), 0)
        self.assertEqual(at.session_state["env_score"], env_score_initial)
        self.assertIsNotNone(at.session_state["analyzed_threat"])


if __name__ == "__main__":
    unittest.main()
