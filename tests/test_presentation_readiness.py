"""
Presentation Readiness and Multimodal Fusion Verification Test Suite.
Validates:
1. One-modality inference (Single-branch contextual assessment).
2. Two-modality fusion (Partial dual-branch adaptive fusion).
3. Three-modality fusion (Full reliability-aware adaptive fusion).
4. Missing/excluded modality handling (Strict safety policy without zero fabrication).
5. Presentation demonstration mode (5 controlled scenarios).
6. Open-Meteo weather integration.
7. Dashboard rendering consistency and diagnostics.
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.fusion.predict import MultimodalThreatPredictor
from src.fusion.adaptive_fusion import ReliabilityAwareAdaptiveFusionEngine
from src.location.campus_map import CampusMapAnalyzer
from app.main import render_overall_risk_panel, render_diagnostics_panel, DEMO_SCENARIOS


class TestPresentationReadiness(unittest.TestCase):
    def setUp(self):
        self.predictor = MultimodalThreatPredictor()
        self.engine = ReliabilityAwareAdaptiveFusionEngine()

    def test_01_one_modality_inference_physiological(self):
        """1. One-modality inference: Physiological only receives 100% weight."""
        res = self.predictor.predict_fusion(
            physiological_stress_score=0.75,
            location_risk_score=None,
            environment_risk_score=None
        )
        self.assertEqual(res["status"], "SUCCESS")
        self.assertEqual(res["fusion_mode"], "SINGLE_BRANCH_FUSION")
        self.assertEqual(res["final_context_score"], 0.75)
        self.assertEqual(res["final_context_score_pct"], 75.0)
        self.assertEqual(res["adaptive_weights"]["physiological"], 1.0)
        self.assertEqual(res["adaptive_weights"]["location"], 0.0)
        self.assertEqual(res["adaptive_weights"]["environment"], 0.0)
        self.assertIn("location", res["excluded_modalities"])
        self.assertIn("environment", res["excluded_modalities"])

    def test_02_one_modality_inference_location(self):
        """1. One-modality inference: Location only receives 100% weight."""
        res = self.predictor.predict_fusion(
            physiological_stress_score=None,
            location_risk_score=0.45,
            environment_risk_score=None
        )
        self.assertEqual(res["status"], "SUCCESS")
        self.assertEqual(res["fusion_mode"], "SINGLE_BRANCH_FUSION")
        self.assertEqual(res["final_context_score"], 0.45)
        self.assertEqual(res["final_context_score_pct"], 45.0)
        self.assertEqual(res["adaptive_weights"]["location"], 1.0)
        self.assertEqual(res["adaptive_weights"]["physiological"], 0.0)
        self.assertIn("physiological", res["excluded_modalities"])

    def test_03_one_modality_inference_environment(self):
        """1. One-modality inference: Environment only receives 100% weight."""
        res = self.predictor.predict_fusion(
            physiological_stress_score=None,
            location_risk_score=None,
            environment_risk_score=0.30
        )
        self.assertEqual(res["status"], "SUCCESS")
        self.assertEqual(res["fusion_mode"], "SINGLE_BRANCH_FUSION")
        self.assertEqual(res["final_context_score"], 0.30)
        self.assertEqual(res["final_context_score_pct"], 30.0)
        self.assertEqual(res["adaptive_weights"]["environment"], 1.0)
        self.assertEqual(res["adaptive_weights"]["physiological"], 0.0)

    def test_04_two_modality_fusion_physio_and_location(self):
        """2. Two-modality fusion: Physiological + Location re-normalizes to 100% (61.54% + 38.46%)."""
        res = self.predictor.predict_fusion(
            physiological_stress_score=0.20,
            location_risk_score=0.60,
            environment_risk_score=None
        )
        self.assertEqual(res["status"], "SUCCESS")
        self.assertEqual(res["fusion_mode"], "PARTIAL_DUAL_BRANCH")
        w_p = res["adaptive_weights"]["physiological"]
        w_l = res["adaptive_weights"]["location"]
        self.assertAlmostEqual(w_p, 0.40 / 0.65, places=4)
        self.assertAlmostEqual(w_l, 0.25 / 0.65, places=4)
        self.assertAlmostEqual(w_p + w_l, 1.0, places=5)
        expected_score = round((0.40 / 0.65) * 0.20 + (0.25 / 0.65) * 0.60, 4)
        self.assertEqual(res["final_context_score"], expected_score)

    def test_05_two_modality_fusion_physio_and_environment(self):
        """2. Two-modality fusion: Physiological + Environmental re-normalizes to 100% (53.33% + 46.67%)."""
        res = self.predictor.predict_fusion(
            physiological_stress_score=0.50,
            location_risk_score=None,
            environment_risk_score=0.20
        )
        self.assertEqual(res["status"], "SUCCESS")
        self.assertEqual(res["fusion_mode"], "PARTIAL_DUAL_BRANCH")
        w_p = res["adaptive_weights"]["physiological"]
        w_e = res["adaptive_weights"]["environment"]
        self.assertAlmostEqual(w_p, 0.40 / 0.75, places=4)
        self.assertAlmostEqual(w_e, 0.35 / 0.75, places=4)
        self.assertAlmostEqual(w_p + w_e, 1.0, places=5)

    def test_06_three_modality_fusion(self):
        """3. Three-modality fusion: All three modalities active with baseline priors (40%, 25%, 35%)."""
        res = self.predictor.predict_fusion(
            physiological_stress_score=0.10,
            location_risk_score=0.20,
            environment_risk_score=0.30
        )
        self.assertEqual(res["status"], "SUCCESS")
        self.assertEqual(res["fusion_mode"], "THREE_BRANCH_FUSION")
        self.assertAlmostEqual(res["adaptive_weights"]["physiological"], 0.40, places=3)
        self.assertAlmostEqual(res["adaptive_weights"]["location"], 0.25, places=3)
        self.assertAlmostEqual(res["adaptive_weights"]["environment"], 0.35, places=3)
        expected = round(0.40 * 0.10 + 0.25 * 0.20 + 0.35 * 0.30, 4)
        self.assertEqual(res["final_context_score"], expected)

    def test_07_all_modalities_missing_insufficient_data(self):
        """4. Missing modality handling: All modalities missing produces INSUFFICIENT DATA without zero risk fabrication."""
        res = self.predictor.predict_fusion(
            physiological_stress_score=None,
            location_risk_score=None,
            environment_risk_score=None
        )
        self.assertEqual(res["status"], "UNAVAILABLE")
        self.assertIsNone(res["final_context_score"])
        self.assertEqual(res["context_classification"], "INSUFFICIENT DATA")
        self.assertEqual(res["fusion_mode"], "INSUFFICIENT_DATA")

    def test_08_presentation_demo_scenarios_execute(self):
        """5. Presentation Demo Mode: All 5 defined scenarios execute successfully and produce expected results."""
        for name, sc in DEMO_SCENARIOS.items():
            res = self.predictor.predict_fusion(
                physiological_stress_score=sc["physio_score"],
                location_risk_score=sc["loc_score"],
                environment_risk_score=sc["env_score"]
            )
            if "INSUFFICIENT DATA" in name or sc["physio_score"] is None:
                self.assertEqual(res["context_classification"], "INSUFFICIENT DATA")
                self.assertIsNone(res["final_context_score"])
            else:
                self.assertEqual(res["status"], "SUCCESS")
                self.assertIsNotNone(res["final_context_score"])
                # Normalized contributing weights must sum to 100%
                active_sum = sum(res["adaptive_weights"][m] for m in ("physiological", "location", "environment"))
                self.assertAlmostEqual(active_sum, 1.0, places=5)

    def test_09_open_meteo_weather_attached_to_environment(self):
        """6. Open-Meteo weather integration: predict_environment_context returns weather_reading block."""
        env_res = self.predictor.predict_environment_context(location="Main Building", hour=14.0, day_of_week="Thursday", is_weekend=0)
        self.assertIn("weather_reading", env_res)
        w = env_res["weather_reading"]
        self.assertIn("status", w)
        self.assertIn("weather_desc", w)

    def test_10_dashboard_panels_render_cleanly(self):
        """7. Dashboard panel rendering: render_overall_risk_panel and render_diagnostics_panel execute without exceptions."""
        with patch("streamlit.html") as mock_html:
            # 1. Active fusion rendering
            res_active = self.predictor.predict_fusion(
                physiological_stress_score=0.9386,
                location_risk_score=0.60,
                environment_risk_score=0.42
            )
            render_overall_risk_panel(res_active)
            render_diagnostics_panel(res_active, "Presentation Demo Mode")
            self.assertTrue(mock_html.called)

            # 2. Insufficient data rendering
            res_insufficient = self.predictor.predict_fusion(None, None, None)
            render_overall_risk_panel(res_insufficient)
            render_diagnostics_panel(res_insufficient, "Real Data Mode")
            self.assertTrue(mock_html.called)


if __name__ == "__main__":
    unittest.main()
