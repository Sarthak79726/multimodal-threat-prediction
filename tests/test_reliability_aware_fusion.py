"""
Unit and Integration Tests for Reliability-Aware Adaptive Multimodal Fusion.
Tests:
1. All three modalities available.
2. Environmental modality unavailable.
3. Physiological modality unavailable.
4. Location modality unavailable.
5. Two modalities unavailable.
6. All modalities unavailable.
7. Reliability equal to 1.00.
8. Stale observations.
9. Invalid observations.
10. Weights always sum to 100% when fusion is available.
11. Displayed contributions sum to the final score (within rounding).
12. All values come from one structured backend result (Section 4 contract).
13. Decision rationale dynamically generated without stale reference-weight text.
14. Dashboard renders the section exactly once without duplicate calls.
15. Learned gating is transparently reported as NOT TRAINED without a valid empirical artifact.
"""

import sys
import math
import unittest
from unittest.mock import patch, MagicMock
import numpy as np
from pathlib import Path

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.fusion.adaptive_fusion import (
    ReliabilityAwareAdaptiveFusionEngine,
    BASELINE_PRIORS,
    FRESHNESS_TAUS_SEC
)
from src.fusion.dynamic_gating import DynamicGatingNetwork
from src.fusion.predict import MultimodalThreatPredictor
from app.main import render_overall_risk_panel


class TestReliabilityAwareFusion(unittest.TestCase):
    def setUp(self):
        self.engine = ReliabilityAwareAdaptiveFusionEngine()
        self.predictor = MultimodalThreatPredictor()

    def test_01_all_three_modalities_available(self):
        """1. All three modalities available: weights equal baseline priors (40/25/35) and sum to 1.0."""
        res = self.engine.fuse(
            p_score=0.20,
            l_score=0.60,
            e_score=0.28,
            physio_age_sec=0.0,
            spatial_age_sec=0.0,
            env_age_sec=0.0
        )
        self.assertEqual(res["status"], "SUCCESS")
        self.assertTrue(res["is_available"])

        weights = res["adaptive_weights"]
        self.assertAlmostEqual(weights["physiological"], 0.40, places=3)
        self.assertAlmostEqual(weights["location"], 0.25, places=3)
        self.assertAlmostEqual(weights["environment"], 0.35, places=3)
        self.assertAlmostEqual(sum([weights["physiological"], weights["location"], weights["environment"]]), 1.0, places=5)

        expected_score = round(0.40 * 0.20 + 0.25 * 0.60 + 0.35 * 0.28, 4)
        self.assertEqual(res["final_context_score"], expected_score)
        self.assertEqual(res["final_context_score_pct"], round(expected_score * 100.0, 2))

        # Contributions sum to final score
        contribs = res["modality_contributions"]
        self.assertAlmostEqual(
            contribs["physiological"] + contribs["location"] + contribs["environment"],
            res["final_context_score"],
            places=4
        )

    def test_02_environmental_modality_unavailable(self):
        """2. Environmental modality unavailable: excluded, remaining weights re-normalize, never 0 for valid."""
        # Scenario from user screenshot: P=0.001 (0.1%), L=0.08 (8.0%), E=None
        res = self.engine.fuse(p_score=0.001, l_score=0.08, e_score=None)
        weights = res["adaptive_weights"]

        # Environmental must be 0% weight and excluded
        self.assertEqual(weights["environment"], 0.0)
        self.assertIn("environment", res["excluded_modalities"])

        # Remaining valid weights re-normalize: 0.40 / 0.65 and 0.25 / 0.65
        w_p_exp = 0.40 / 0.65
        w_l_exp = 0.25 / 0.65
        self.assertAlmostEqual(weights["physiological"], w_p_exp, places=4)
        self.assertAlmostEqual(weights["location"], w_l_exp, places=4)
        self.assertGreater(weights["physiological"], 0.0, "Valid physiological modality must never have 0 weight")
        self.assertGreater(weights["location"], 0.0, "Valid location modality must never have 0 weight")

        # Sum of unique modality weights must be exactly 1.0
        w_sum = weights["physiological"] + weights["location"] + weights["environment"]
        self.assertAlmostEqual(w_sum, 1.0, places=5)

        # Expected score: w_p * 0.001 + w_l * 0.08 = 0.0314 (3.14%), NOT 10.78%
        expected_score = round(w_p_exp * 0.001 + w_l_exp * 0.08, 4)
        self.assertEqual(res["final_context_score"], expected_score)
        self.assertEqual(res["final_context_score_pct"], 3.14)
        self.assertNotEqual(res["final_context_score_pct"], 10.78)

        # Contributions sum to final score
        contribs = res["modality_contributions"]
        self.assertAlmostEqual(contribs["physiological"] + contribs["location"], res["final_context_score"], places=4)

    def test_03_physiological_modality_unavailable(self):
        """3. Physiological modality unavailable: excluded, remaining weights re-normalize to 1.0."""
        res = self.engine.fuse(p_score=None, l_score=0.50, e_score=0.30)
        weights = res["adaptive_weights"]

        self.assertEqual(weights["physiological"], 0.0)
        self.assertIn("physiological", res["excluded_modalities"])

        # Location and Environment re-normalize: 0.25 / 0.60 and 0.35 / 0.60
        w_l_exp = 0.25 / 0.60
        w_e_exp = 0.35 / 0.60
        self.assertAlmostEqual(weights["location"], w_l_exp, places=4)
        self.assertAlmostEqual(weights["environment"], w_e_exp, places=4)
        self.assertAlmostEqual(weights["location"] + weights["environment"], 1.0, places=5)

        expected_score = round(w_l_exp * 0.50 + w_e_exp * 0.30, 4)
        self.assertEqual(res["final_context_score"], expected_score)

        # Contributions sum to final score
        contribs = res["modality_contributions"]
        self.assertAlmostEqual(contribs["location"] + contribs["environment"], res["final_context_score"], places=4)

    def test_04_location_modality_unavailable(self):
        """4. Location modality unavailable: excluded, remaining weights re-normalize to 1.0."""
        res = self.engine.fuse(p_score=0.40, l_score=None, e_score=0.30)
        weights = res["adaptive_weights"]

        self.assertEqual(weights["location"], 0.0)
        self.assertIn("location", res["excluded_modalities"])

        # Physiological and Environment re-normalize: 0.40 / 0.75 and 0.35 / 0.75
        w_p_exp = 0.40 / 0.75
        w_e_exp = 0.35 / 0.75
        self.assertAlmostEqual(weights["physiological"], w_p_exp, places=4)
        self.assertAlmostEqual(weights["environment"], w_e_exp, places=4)
        self.assertAlmostEqual(weights["physiological"] + weights["environment"], 1.0, places=5)

        expected_score = round(w_p_exp * 0.40 + w_e_exp * 0.30, 4)
        self.assertEqual(res["final_context_score"], expected_score)

        # Contributions sum to final score
        contribs = res["modality_contributions"]
        self.assertAlmostEqual(contribs["physiological"] + contribs["environment"], res["final_context_score"], places=4)

    def test_05_two_modalities_unavailable(self):
        """5. Two modalities unavailable: single available modality gets 100% weight."""
        res = self.engine.fuse(p_score=None, l_score=0.55, e_score=None)
        weights = res["adaptive_weights"]

        self.assertEqual(weights["location"], 1.0)
        self.assertEqual(weights["physiological"], 0.0)
        self.assertEqual(weights["environment"], 0.0)
        self.assertIn("physiological", res["excluded_modalities"])
        self.assertIn("environment", res["excluded_modalities"])

        self.assertEqual(res["final_context_score"], 0.55)
        self.assertEqual(res["final_context_score_pct"], 55.0)

        # Contributions sum to final score
        contribs = res["modality_contributions"]
        self.assertAlmostEqual(contribs["location"], res["final_context_score"], places=4)

    def test_06_all_modalities_unavailable(self):
        """6. All modalities unavailable: returns UNAVAILABLE with None score (never 0 risk)."""
        res = self.engine.fuse(p_score=None, l_score=None, e_score=None)

        self.assertEqual(res["status"], "UNAVAILABLE")
        self.assertIsNone(res["final_context_score"])
        self.assertIsNone(res["final_context_score_pct"])
        self.assertIsNone(res["overall_context_index"])
        self.assertEqual(res["fusion_mode"], "UNAVAILABLE")
        self.assertEqual(res["context_classification"], "UNAVAILABLE")

        # Weights are 0.0
        self.assertEqual(res["adaptive_weights"]["physiological"], 0.0)
        self.assertEqual(res["adaptive_weights"]["location"], 0.0)
        self.assertEqual(res["adaptive_weights"]["environment"], 0.0)

        # Predictor returns UNAVAILABLE with None score
        pred_res = self.predictor.predict_fusion(
            physiological_stress_score=None,
            location_risk_score=None,
            environment_risk_score=None
        )
        self.assertEqual(pred_res["status"], "UNAVAILABLE")
        self.assertIsNone(pred_res["final_context_score"])
        self.assertIsNone(pred_res["threat_probability"])

    def test_07_reliability_equal_to_one(self):
        """7. Reliability equal to 1.00: weights match baseline priors, never 0 for valid modalities."""
        res = self.engine.fuse(p_score=0.10, l_score=0.20, e_score=0.30)
        reliabilities = res["modality_reliabilities"]

        self.assertEqual(reliabilities["physiological"]["composite_q"], 1.0)
        self.assertEqual(reliabilities["location"]["composite_q"], 1.0)
        self.assertEqual(reliabilities["environment"]["composite_q"], 1.0)

        weights = res["adaptive_weights"]
        self.assertAlmostEqual(weights["physiological"], 0.40, places=3)
        self.assertAlmostEqual(weights["location"], 0.25, places=3)
        self.assertAlmostEqual(weights["environment"], 0.35, places=3)
        self.assertGreater(weights["physiological"], 0.0)
        self.assertGreater(weights["location"], 0.0)
        self.assertGreater(weights["environment"], 0.0)

    def test_08_stale_observations(self):
        """8. Stale observations: weight decays exponentially via exp(-age/tau)."""
        tau_l = FRESHNESS_TAUS_SEC["spatial"]  # 300s
        res_fresh = self.engine.fuse(p_score=0.30, l_score=0.70, e_score=0.40, spatial_age_sec=0.0)
        res_stale = self.engine.fuse(p_score=0.30, l_score=0.70, e_score=0.40, spatial_age_sec=tau_l * 4.0)

        w_l_fresh = res_fresh["adaptive_weights"]["location"]
        w_l_stale = res_stale["adaptive_weights"]["location"]

        self.assertLess(w_l_stale, w_l_fresh)
        self.assertLess(w_l_stale, 0.02)  # exp(-4) ~ 0.0183 -> weight drops below 2%
        self.assertGreater(res_stale["adaptive_weights"]["physiological"], res_fresh["adaptive_weights"]["physiological"])

        # Sum still equals 1.0
        w_sum = res_stale["adaptive_weights"]["physiological"] + res_stale["adaptive_weights"]["location"] + res_stale["adaptive_weights"]["environment"]
        self.assertAlmostEqual(w_sum, 1.0, places=5)

    def test_09_invalid_observations(self):
        """9. Invalid observations: NaN / Inf receives 0 weight and remaining modalities re-normalize."""
        res = self.engine.fuse(p_score=float("nan"), l_score=0.50, e_score=0.30)
        weights = res["adaptive_weights"]

        self.assertEqual(weights["physiological"], 0.0)
        self.assertIn("physiological", res["excluded_modalities"])
        self.assertAlmostEqual(weights["location"] + weights["environment"], 1.0, places=5)

    def test_10_structured_fusion_result_contract(self):
        """10. Structured fusion result contains all 9 required keys (Section 4 contract)."""
        pred_res = self.predictor.predict_fusion(
            physiological_stress_score=0.001,
            location_risk_score=0.08,
            environment_risk_score=None
        )

        required_keys = [
            "final_context_score",
            "fusion_mode",
            "modality_scores",
            "modality_reliability",
            "modality_freshness",
            "adaptive_weights",
            "excluded_modalities",
            "decision_rationale",
            "meta_learner_status"
        ]
        for key in required_keys:
            self.assertIn(key, pred_res, f"Missing required Section 4 key: {key}")

        # Aliases for robustness
        self.assertIn("spatial", pred_res["adaptive_weights"])
        self.assertIn("environmental", pred_res["adaptive_weights"])

    def test_11_decision_rationale_dynamically_generated(self):
        """11. Decision rationale dynamically reports actual scores, weights, excluded modalities, and final score."""
        pred_res = self.predictor.predict_fusion(
            physiological_stress_score=0.001,
            location_risk_score=0.08,
            environment_risk_score=None
        )
        rationale = pred_res["decision_rationale"]

        # Must report actual scores
        self.assertIn("0.1%", rationale)
        self.assertIn("8.0%", rationale)
        # Must report actual weights
        self.assertIn("61.5%", rationale)
        self.assertIn("38.5%", rationale)
        # Must report excluded modalities
        self.assertIn("Environment", rationale)
        # Must report final score
        self.assertIn("3.14%", rationale)
        # Must NOT report old 10.78% or reference weights
        self.assertNotIn("10.78%", rationale)

    def test_12_dashboard_renders_single_panel_without_duplicate(self):
        """12. Dashboard render_overall_risk_panel renders exactly once (no duplicate st.html)."""
        pred_res = self.predictor.predict_fusion(
            physiological_stress_score=0.001,
            location_risk_score=0.08,
            environment_risk_score=None
        )
        with patch("streamlit.html") as mock_st_html:
            render_overall_risk_panel(pred_res)
            # Verify st.html was called exactly once
            self.assertEqual(mock_st_html.call_count, 1, "render_overall_risk_panel must call st.html exactly once")

    def test_13_learned_gating_never_reported_as_trained_without_artifact(self):
        """13. Learned gating is transparently reported as NOT TRAINED without a valid empirical artifact."""
        gating = DynamicGatingNetwork()
        self.assertFalse(gating.is_trained)

        dummy_x = np.zeros((1, 15), dtype=np.float64)
        weights_dict = gating.predict_weights_dict(dummy_x)
        self.assertEqual(weights_dict["learned_gating_status"], "LEARNED GATING: NOT TRAINED")
        self.assertFalse(weights_dict["is_trained"])

        pred_res = self.predictor.predict_fusion(0.5, 0.5, 0.5)
        self.assertEqual(pred_res["learned_gating_status"], "LEARNED GATING: NOT TRAINED")
        self.assertEqual(pred_res["meta_learner_status"], "LEARNED GATING: NOT TRAINED")
        self.assertFalse(pred_res["is_meta_learner_trained"])


if __name__ == "__main__":
    unittest.main()

