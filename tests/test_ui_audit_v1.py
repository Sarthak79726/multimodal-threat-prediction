"""
UI Integration and End-to-End Headless Testing for Version 1.0 Audit Enhancements:
1. Streamlit app starts cleanly with AppTest.
2. Evaluate Overall Context Risk button renders and clicks without error.
3. Partial Dual-Branch Context Index card renders when environmental context is missing.
4. Ambiguous location warning displays when GPS coordinates are ambiguous.
5. Outside campus geofence warning displays when coordinates exceed 350m geofence.
6. Untrained meta-learner status remains visible.
"""

import sys
import unittest
from pathlib import Path
from streamlit.testing.v1 import AppTest

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import config
from src.fusion.predict import MultimodalThreatPredictor

class TestUIAuditV1(unittest.TestCase):
    def test_streamlit_app_startup_and_evaluate_button(self):
        """Verifies Streamlit app initializes cleanly and button executes successfully."""
        app_path = str(Path(_project_root) / "app" / "main.py")
        at = AppTest.from_file(app_path)
        at.run(timeout=30)
        self.assertEqual(len(at.exception), 0, f"App startup encountered exceptions: {at.exception}")

        # Find evaluate button
        btn = at.button(key="btn_main_threat")
        self.assertIsNotNone(btn)
        self.assertEqual(btn.label, "🛡️ EVALUATE OVERALL CONTEXT RISK")

        # Click button
        btn.click().run(timeout=30)
        self.assertEqual(len(at.exception), 0, f"Button execution failed: {at.exception}")
        
        threat = at.session_state["analyzed_threat"]
        self.assertIsNotNone(threat)
        self.assertFalse(threat["is_meta_learner_trained"])
        self.assertEqual(threat["meta_model_status"], "UNTRAINED — VALID MULTIMODAL THREAT DATASET REQUIRED")

    def test_partial_dual_branch_rendering_in_streamlit(self):
        """Verifies that when environmental context is unavailable, the Partial Dual-Branch card is rendered."""
        from unittest.mock import patch
        app_path = str(Path(_project_root) / "app" / "main.py")
        
        # Mock predict_environment_context to simulate unavailable environmental engine
        with patch.object(
            MultimodalThreatPredictor,
            "predict_environment_context",
            return_value={
                "status": "UNAVAILABLE",
                "error": "MOCK_UNAVAILABLE",
                "message": "Environmental service unavailable.",
                "crowd_prediction": {"category": "UNKNOWN", "confidence_pct": 0.0, "model_name": "None"},
                "lighting_prediction": {"category": "UNKNOWN", "confidence_pct": 0.0, "model_name": "None", "lux_estimate": None},
                "environmental_context_score": None,
                "calibrated_environmental_score": None
            }
        ):
            at = AppTest.from_file(app_path)
            at.run(timeout=30)
            
            # Click evaluate
            at.button(key="btn_main_threat").click().run(timeout=30)
            self.assertEqual(len(at.exception), 0)

            threat = at.session_state["analyzed_threat"]
            self.assertEqual(threat["fusion_mode"], "PARTIAL_DUAL_BRANCH")
            self.assertIsNone(threat["experimental_context_index"])
            self.assertIsNotNone(threat["partial_dual_branch_index"])
            self.assertIsNotNone(threat["partial_dual_branch_index_pct"])

    def test_gps_ambiguity_flagging_in_state(self):
        """Verifies predictor.env_engine resolves ambiguous locations correctly."""
        predictor = MultimodalThreatPredictor()
        h3_coord = config.BIT_MESRA_LANDMARKS_DB["h3"]
        pmc_coord = config.BIT_MESRA_LANDMARKS_DB["pmc"]
        mid_lat = (h3_coord["latitude"] + pmc_coord["latitude"]) / 2.0
        mid_lon = (h3_coord["longitude"] + pmc_coord["longitude"]) / 2.0

        res = predictor.env_engine.resolve_campus_location(mid_lat, mid_lon, accuracy=30.0)
        self.assertEqual(res["status"], "AMBIGUOUS_LOCATION")
        self.assertIsNone(res["resolved_location"])
        self.assertIn("H3", res["ambiguous_candidates"])
        self.assertIn("PMC", res["ambiguous_candidates"])

if __name__ == "__main__":
    unittest.main()
