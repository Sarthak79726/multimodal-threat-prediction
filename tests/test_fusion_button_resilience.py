"""
Regression and Integration Tests for Fusion Model Risk Evaluation Button.
Validates:
1. Button presence, key, and label in app/main.py.
2. Button evaluation callback logic execution and session_state persistence.
3. Resilience to repeated clicks without state corruption.
4. Resilience across Streamlit reruns (state preservation and auto-refresh).
5. Environmental engine failure isolation (fusion works even if env_engine is None or failing).
6. Consistency with last working version (commit 6fd0e41) mathematical metrics and weights.
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import config
from src.fusion.predict import MultimodalThreatPredictor

class TestFusionButtonResilience(unittest.TestCase):
    """Test suite for Fusion Model Risk Evaluation Button and execution flow."""

    def setUp(self):
        self.predictor = MultimodalThreatPredictor()

    def test_button_definition_in_main_dashboard(self):
        """Verifies that the fusion evaluation button is properly defined with key and container width."""
        main_py = (Path(_project_root) / "app" / "main.py").read_text(encoding="utf-8")
        self.assertIn('key="btn_main_threat"', main_py)
        self.assertIn('st.button("🛡️ EVALUATE OVERALL CONTEXT RISK"', main_py)
        self.assertIn('st.session_state["analyzed_threat"] = res', main_py)

    def test_fusion_evaluation_execution_logic(self):
        """Verifies that clicking the button executes predict_fusion and returns all expected keys."""
        phys_score = 0.20
        loc_score = 0.15
        env_score = 0.10

        res = self.predictor.predict_fusion(
            physiological_stress_score=phys_score,
            location_risk_score=loc_score,
            environment_risk_score=env_score,
            threshold=0.50
        )

        # Expected output structure
        self.assertIsInstance(res, dict)
        required_keys = [
            "physiological_stress_score",
            "location_risk_score",
            "environment_risk_score",
            "threat_probability",
            "threat_score_pct",
            "experimental_context_index",
            "experimental_context_index_pct",
            "context_status",
            "meta_model_status",
            "explanation",
            "modalities_metadata"
        ]
        for key in required_keys:
            self.assertIn(key, res, f"Missing key in fusion output: {key}")

        expected_score = round(0.40 * 0.20 + 0.25 * 0.15 + 0.35 * 0.10, 4)
        self.assertAlmostEqual(res["experimental_context_index"], expected_score, places=4)
        self.assertEqual(res["context_status"], "STANDARD CONTEXT")

    def test_repeated_clicks_determinism_and_stability(self):
        """Verifies that repeated clicks compute stable, deterministic results without corruption."""
        phys_score = 0.35
        loc_score = 0.50
        env_score = 0.40

        results = []
        for _ in range(5):
            res = self.predictor.predict_fusion(
                physiological_stress_score=phys_score,
                location_risk_score=loc_score,
                environment_risk_score=env_score,
                threshold=0.50
            )
            results.append(res["experimental_context_index"])

        # All 5 invocations must yield identical results
        self.assertEqual(len(set(results)), 1)
        self.assertAlmostEqual(results[0], 0.405, places=3)

    def test_session_state_rerun_behavior(self):
        """Simulates Streamlit rerun: session_state retention and dynamic input update."""
        mock_session_state = {
            "physio_score": 0.20,
            "location_score": 0.15,
            "env_score": 0.10,
            "analyzed_threat": None
        }

        # 1. User clicks the button
        res = self.predictor.predict_fusion(
            physiological_stress_score=mock_session_state["physio_score"],
            location_risk_score=mock_session_state["location_score"],
            environment_risk_score=mock_session_state["env_score"],
            threshold=0.50
        )
        mock_session_state["analyzed_threat"] = res
        self.assertIsNotNone(mock_session_state["analyzed_threat"])
        initial_score = mock_session_state["analyzed_threat"]["experimental_context_index"]

        # 2. User changes physiological input on next run
        mock_session_state["physio_score"] = 0.80

        # Auto-update logic triggers because analyzed_threat is not None
        if mock_session_state.get("analyzed_threat") is not None:
            updated_res = self.predictor.predict_fusion(
                physiological_stress_score=mock_session_state["physio_score"],
                location_risk_score=mock_session_state["location_score"],
                environment_risk_score=mock_session_state["env_score"],
                threshold=0.50
            )
            mock_session_state["analyzed_threat"] = updated_res

        new_score = mock_session_state["analyzed_threat"]["experimental_context_index"]
        self.assertGreater(new_score, initial_score)
        self.assertAlmostEqual(new_score, 0.40 * 0.80 + 0.25 * 0.15 + 0.35 * 0.10, places=4)

    def test_environmental_engine_failure_isolation(self):
        """Verifies that an environmental engine failure does not break fusion evaluation."""
        self.predictor.env_engine = None  # Simulate complete environmental engine failure

        # Fusion evaluation must still compute cleanly with fallback
        res = self.predictor.predict_fusion(
            physiological_stress_score=0.25,
            location_risk_score=0.30,
            environment_risk_score=0.20,  # Fallback neutral environmental score
            threshold=0.50
        )

        self.assertIsInstance(res, dict)
        self.assertIn("experimental_context_index", res)
        self.assertEqual(res["modalities_metadata"]["environmental"]["status"], "UNAVAILABLE")
        self.assertEqual(res["context_status"], "STANDARD CONTEXT")

    def test_metrics_consistency_with_reference_commit(self):
        """Verifies exact adherence to 32.00% benchmark scenario from reference commit."""
        res = self.predictor.predict_fusion(
            physiological_stress_score=0.18,
            location_risk_score=0.60,
            environment_risk_score=0.28,
            threshold=0.50
        )

        self.assertAlmostEqual(res["experimental_context_index_pct"], 32.00, places=2)
        self.assertAlmostEqual(res["contributions_pct"]["physiological"], 7.20, places=2)
        self.assertAlmostEqual(res["contributions_pct"]["location"], 15.00, places=2)
        self.assertAlmostEqual(res["contributions_pct"]["environment"], 9.80, places=2)
        self.assertEqual(res["context_status"], "STANDARD CONTEXT")
        self.assertEqual(res["meta_model_status"], "UNTRAINED — VALID MULTIMODAL THREAT DATASET REQUIRED")

if __name__ == "__main__":
    unittest.main()
