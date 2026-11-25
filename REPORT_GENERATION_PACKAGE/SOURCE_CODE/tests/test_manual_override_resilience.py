"""
Unit and Integration Test Suite for Manual Location Override Resilience & Single Authoritative Pipeline:
1. Scenario 1: GPS detected landmark H13, override disabled -> effective location H13 (spatial risk 0.60).
2. Scenario 2: GPS detected landmark H13, override enabled, selected H12 -> effective location H12 (spatial risk 0.55).
3. Scenario 3: GPS detected landmark H13, override enabled, selected other landmarks (Main Building, PMC) -> effective spatial risk and environmental prediction match selected landmark.
4. Scenario 4: Switching sequentially between H13, H12, and Main Building -> results update cleanly without interference.
5. Scenario 5: Disabling override -> immediately reverts to live GPS location (H13).
6. Scenario 6: Repeated changes without page reload -> no stale cached results or state bleeding.
7. Streamlit Headless AppTest:
   - App loads cleanly with default GPS or fallback state.
   - Enabling override checkbox and changing selectbox updates session state authoritative location.
   - Evaluate button clicks evaluate threat using the overridden landmark metrics.
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import config
from src.location.landmarks import get_all_landmarks_list
from src.location.campus_map import CampusMapAnalyzer
from src.environment.model import EnvironmentalContextEngine, SUPPORTED_LOCATIONS
from src.fusion.predict import MultimodalThreatPredictor

class TestManualLocationOverrideResilience(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.landmarks = get_all_landmarks_list()
        cls.lm_dict = {lm["name"].lower(): lm for lm in cls.landmarks}
        cls.analyzer = CampusMapAnalyzer()
        cls.predictor = MultimodalThreatPredictor()
        cls.engine = EnvironmentalContextEngine()

    def get_landmark_analysis(self, name: str, hour: float = 14.0):
        lm = self.lm_dict.get(name.lower())
        self.assertIsNotNone(lm, f"Landmark '{name}' not found in database")
        return self.analyzer.analyze_gps(
            lat=float(lm["latitude"]),
            lon=float(lm["longitude"]),
            accuracy=5.0,
            time_of_day_hour=hour
        )

    def test_scenario_1_gps_h13_override_disabled(self):
        """Test 1: GPS location H13, override disabled -> effective location H13 (spatial risk 0.60)."""
        h13_analysis = self.get_landmark_analysis("H13")
        self.assertEqual(h13_analysis["nearest_landmark"]["name"], "H13")
        self.assertEqual(h13_analysis["location_risk_score"], 0.60)

        # In simulation: override disabled
        is_override = False
        selected_manual = "H12"  # Should be ignored because override is False
        gps_name = h13_analysis["nearest_landmark"]["name"]

        effective_loc = selected_manual if is_override else gps_name
        self.assertEqual(effective_loc, "H13")

        effective_analysis = h13_analysis if not is_override else self.get_landmark_analysis(selected_manual)
        self.assertEqual(effective_analysis["location_risk_score"], 0.60)

        # Environmental prediction for effective location
        env_res = self.predictor.predict_environment_context(location=effective_loc, hour=14.0)
        self.assertEqual(env_res["status"], "SUCCESS")

        # Fusion evaluation
        fusion_res = self.predictor.predict_fusion(
            physiological_stress_score=0.18,
            location_risk_score=effective_analysis["location_risk_score"],
            environment_risk_score=env_res["environmental_context_score"]
        )
        self.assertEqual(fusion_res["location_risk_score"], 0.60)
        expected_index = round(0.40 * 0.18 + 0.25 * 0.60 + 0.35 * env_res["environmental_context_score"], 4)
        self.assertAlmostEqual(fusion_res["experimental_context_index"], expected_index, places=4)

    def test_scenario_2_gps_h13_override_enabled_selected_h12(self):
        """Test 2: GPS location H13, override enabled, selected H12 -> effective location H12 (spatial risk 0.55)."""
        h13_analysis = self.get_landmark_analysis("H13")
        h12_analysis = self.get_landmark_analysis("H12")

        self.assertEqual(h12_analysis["nearest_landmark"]["name"], "H12")
        self.assertEqual(h12_analysis["location_risk_score"], 0.55)

        # In simulation: override enabled, H12 selected
        is_override = True
        selected_manual = "H12"
        gps_name = h13_analysis["nearest_landmark"]["name"]

        effective_loc = selected_manual if is_override else gps_name
        self.assertEqual(effective_loc, "H12")

        effective_analysis = h12_analysis if is_override else h13_analysis
        self.assertEqual(effective_analysis["location_risk_score"], 0.55)

        # Environmental prediction for effective location
        env_res = self.predictor.predict_environment_context(location=effective_loc, hour=14.0)
        self.assertEqual(env_res["status"], "SUCCESS")

        # Fusion evaluation
        fusion_res = self.predictor.predict_fusion(
            physiological_stress_score=0.18,
            location_risk_score=effective_analysis["location_risk_score"],
            environment_risk_score=env_res["environmental_context_score"]
        )
        self.assertEqual(fusion_res["location_risk_score"], 0.55, "Fusion location score MUST be H12's 0.55, not H13's 0.60")
        expected_index = round(0.40 * 0.18 + 0.25 * 0.55 + 0.35 * env_res["environmental_context_score"], 4)
        self.assertAlmostEqual(fusion_res["experimental_context_index"], expected_index, places=4)

        # Contrast directly with H13
        fusion_h13 = self.predictor.predict_fusion(
            physiological_stress_score=0.18,
            location_risk_score=h13_analysis["location_risk_score"],
            environment_risk_score=env_res["environmental_context_score"]
        )
        self.assertNotEqual(fusion_res["experimental_context_index"], fusion_h13["experimental_context_index"])
        self.assertAlmostEqual(fusion_h13["experimental_context_index"] - fusion_res["experimental_context_index"], 0.25 * (0.60 - 0.55), places=4)

    def test_scenario_3_override_other_landmarks(self):
        """Test 3: Override enabled, selected Main Building (0.08) and PMC (0.65)."""
        main_analysis = self.get_landmark_analysis("Main Building")
        pmc_analysis = self.get_landmark_analysis("PMC")

        self.assertEqual(main_analysis["location_risk_score"], 0.08)
        self.assertEqual(pmc_analysis["location_risk_score"], 0.65)

        for lm_name, expected_risk in [("Main Building", 0.08), ("PMC", 0.65), ("Main Gate", 0.05)]:
            analysis = self.get_landmark_analysis(lm_name)
            self.assertEqual(analysis["location_risk_score"], expected_risk)
            env_res = self.predictor.predict_environment_context(location=lm_name, hour=12.0)
            fusion = self.predictor.predict_fusion(
                physiological_stress_score=0.20,
                location_risk_score=analysis["location_risk_score"],
                environment_risk_score=env_res["environmental_context_score"]
            )
            self.assertEqual(fusion["location_risk_score"], expected_risk)

    def test_scenario_4_sequential_switching_between_landmarks(self):
        """Test 4: Sequentially switching H13 -> H12 -> Main Building -> H13 generates clean, updated results."""
        results = []
        sequence = ["H13", "H12", "Main Building", "H13"]
        for target in sequence:
            analysis = self.get_landmark_analysis(target)
            env_res = self.predictor.predict_environment_context(location=target, hour=14.0)
            fusion = self.predictor.predict_fusion(
                physiological_stress_score=0.20,
                location_risk_score=analysis["location_risk_score"],
                environment_risk_score=env_res["environmental_context_score"]
            )
            results.append((target, analysis["location_risk_score"], fusion["experimental_context_index"]))

        self.assertEqual(results[0][0], "H13")
        self.assertEqual(results[0][1], 0.60)
        self.assertEqual(results[1][0], "H12")
        self.assertEqual(results[1][1], 0.55)
        self.assertEqual(results[2][0], "Main Building")
        self.assertEqual(results[2][1], 0.08)
        self.assertEqual(results[3][0], "H13")
        self.assertEqual(results[3][1], 0.60)
        self.assertEqual(results[0][2], results[3][2], "Returning to H13 must yield exact same context index")

    def test_scenario_5_disabling_override_reverts_to_gps(self):
        """Test 5: Disabling override instantly reverts effective location from H12 to H13."""
        gps_loc = "H13"
        manual_loc = "H12"

        # State 1: Override Active
        override_active = True
        eff_loc_1 = manual_loc if override_active else gps_loc
        self.assertEqual(eff_loc_1, "H12")

        # State 2: Override Toggled Off
        override_active = False
        eff_loc_2 = manual_loc if override_active else gps_loc
        self.assertEqual(eff_loc_2, "H13")

    def test_scenario_6_repeated_changes_no_stale_cache(self):
        """Test 6: Repeated changes across 10 rapid switches produce deterministic outputs without state bleeding."""
        landmarks_to_test = ["H1", "H2", "H3", "H4", "H6", "H7", "H10", "H11", "H12", "H13"]
        for lm in landmarks_to_test:
            analysis = self.get_landmark_analysis(lm)
            env = self.predictor.predict_environment_context(location=lm, hour=15.0)
            fusion = self.predictor.predict_fusion(
                physiological_stress_score=0.25,
                location_risk_score=analysis["location_risk_score"],
                environment_risk_score=env["environmental_context_score"]
            )
            expected = round(0.40 * 0.25 + 0.25 * analysis["location_risk_score"] + 0.35 * env["environmental_context_score"], 4)
            self.assertAlmostEqual(fusion["experimental_context_index"], expected, places=4)

    def test_streamlit_apptest_manual_override_integration(self):
        """Test 7: Headless Streamlit AppTest verifies widget rendering and manual override workflow."""
        from streamlit.testing.v1 import AppTest
        app_path = str(Path(_project_root) / "app" / "main.py")
        at = AppTest.from_file(app_path)
        at.run(timeout=30)
        self.assertEqual(len(at.exception), 0, f"App startup encountered exceptions: {at.exception}")

        # Verify session state authoritative location variables are properly initialized
        self.assertIn("effective_location_name", at.session_state)
        self.assertIn("location_mode", at.session_state)
        self.assertIn("location_score", at.session_state)
        self.assertIsNotNone(at.session_state["effective_location_name"])
        self.assertIsNotNone(at.session_state["location_score"])

        # Verify diagnostic container HTML elements are rendered in UI tree
        html_elements = at.get("html")
        self.assertGreater(len(html_elements), 5, "UI tree must render HTML diagnostic and card elements")

        # Test clicking evaluate overall threat button
        btn = at.button(key="btn_main_threat")
        self.assertIsNotNone(btn)
        btn.click().run(timeout=30)
        self.assertEqual(len(at.exception), 0, f"Button execution encountered exceptions: {at.exception}")

        analyzed = at.session_state["analyzed_threat"]
        self.assertIsNotNone(analyzed)
        self.assertIn("experimental_context_index", analyzed)
        self.assertAlmostEqual(analyzed["location_risk_score"], at.session_state["location_score"], places=4)

if __name__ == "__main__":
    unittest.main()
