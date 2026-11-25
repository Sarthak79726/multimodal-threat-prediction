"""
Unit and Integration Tests for Reliability-Aware Adaptive Multimodal Fusion.
Tests:
1. All three modalities valid.
2. Only physiological input available.
3. Only location input available.
4. Only environmental input available.
5. Two modalities available and one excluded.
6. All modalities excluded (INSUFFICIENT DATA).
7. Stale modality input.
8. Invalid modality input (NaN, inf, out-of-range).
9. Zero reliability or zero total effective weight.
10. Weight normalization and contribution reconciliation.
11. Dashboard values matching the authoritative fusion result.
12. Missing inputs never producing a fabricated context score.
13. Authoritative result contract (Section 2 & 4).
14. Dynamically generated decision rationale matching calculation.
15. Single panel rendering without duplicate DOM output.
16. Learned gating transparently reported as NOT TRAINED.
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

    def test_01_all_three_modalities_valid(self):
        """1. All three modalities valid: weights match baseline priors (40/25/35), sum to 100%, and contribute correctly."""
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
        self.assertEqual(res["fusion_mode"], "THREE_BRANCH_FUSION")

        weights = res["adaptive_weights"]
        self.assertAlmostEqual(weights["physiological"], 0.40, places=3)
        self.assertAlmostEqual(weights["location"], 0.25, places=3)
        self.assertAlmostEqual(weights["environment"], 0.35, places=3)
        self.assertAlmostEqual(sum([weights["physiological"], weights["location"], weights["environment"]]), 1.0, places=5)

        expected_score = round(0.40 * 0.20 + 0.25 * 0.60 + 0.35 * 0.28, 4)
        self.assertEqual(res["final_context_score"], expected_score)
        self.assertEqual(res["final_context_score_pct"], round(expected_score * 100.0, 2))

        # Check authoritative modalities dict
        mods = res["modalities"]
        for m_name in ("physiological", "location", "environment"):
            self.assertIn(m_name, mods)
            self.assertFalse(mods[m_name]["is_excluded"])
            self.assertEqual(mods[m_name]["exclusion_reason"], "")

        # Contributions sum to final score
        contribs = res["modality_contributions"]
        self.assertAlmostEqual(
            contribs["physiological"] + contribs["location"] + contribs["environment"],
            res["final_context_score"],
            places=4
        )

    def test_02_only_physiological_available(self):
        """2. Only physiological input available: gets 100% normalized weight, other modalities excluded with clear reasons."""
        res = self.engine.fuse(p_score=0.35, l_score=None, e_score=None)
        self.assertEqual(res["status"], "SUCCESS")
        self.assertEqual(res["fusion_mode"], "SINGLE_BRANCH_FUSION")
        self.assertEqual(res["final_context_score"], 0.35)
        self.assertEqual(res["final_context_score_pct"], 35.0)

        weights = res["adaptive_weights"]
        self.assertEqual(weights["physiological"], 1.0)
        self.assertEqual(weights["location"], 0.0)
        self.assertEqual(weights["environment"], 0.0)

        mods = res["modalities"]
        self.assertFalse(mods["physiological"]["is_excluded"])
        self.assertTrue(mods["location"]["is_excluded"])
        self.assertTrue(mods["environment"]["is_excluded"])
        self.assertIn("Missing", mods["location"]["exclusion_reason"])
        self.assertIn("offline or unavailable", mods["environment"]["exclusion_reason"])

    def test_03_only_location_available(self):
        """3. Only location input available: gets 100% normalized weight, physiological and environmental excluded."""
        res = self.engine.fuse(p_score=None, l_score=0.55, e_score=None)
        self.assertEqual(res["status"], "SUCCESS")
        self.assertEqual(res["fusion_mode"], "SINGLE_BRANCH_FUSION")
        self.assertEqual(res["final_context_score"], 0.55)
        self.assertEqual(res["final_context_score_pct"], 55.0)

        weights = res["adaptive_weights"]
        self.assertEqual(weights["location"], 1.0)
        self.assertEqual(weights["physiological"], 0.0)
        self.assertEqual(weights["environment"], 0.0)

        mods = res["modalities"]
        self.assertTrue(mods["physiological"]["is_excluded"])
        self.assertFalse(mods["location"]["is_excluded"])
        self.assertTrue(mods["environment"]["is_excluded"])

    def test_04_only_environmental_available(self):
        """4. Only environmental input available: gets 100% normalized weight, biometrics and location excluded."""
        res = self.engine.fuse(p_score=None, l_score=None, e_score=0.72)
        self.assertEqual(res["status"], "SUCCESS")
        self.assertEqual(res["fusion_mode"], "SINGLE_BRANCH_FUSION")
        self.assertEqual(res["final_context_score"], 0.72)
        self.assertEqual(res["final_context_score_pct"], 72.0)

        weights = res["adaptive_weights"]
        self.assertEqual(weights["environment"], 1.0)
        self.assertEqual(weights["physiological"], 0.0)
        self.assertEqual(weights["location"], 0.0)

        mods = res["modalities"]
        self.assertTrue(mods["physiological"]["is_excluded"])
        self.assertTrue(mods["location"]["is_excluded"])
        self.assertFalse(mods["environment"]["is_excluded"])

    def test_05_two_modalities_available_one_excluded(self):
        """5. Two modalities available and one excluded: weights re-normalize, excluded modality has 0% weight and reason."""
        # P=0.001 (0.1%), L=0.08 (8.0%), E=None
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

        # Sum of contributing weights must equal 1.0
        self.assertAlmostEqual(weights["physiological"] + weights["location"], 1.0, places=5)

        # Score matches formula: w_p * 0.001 + w_l * 0.08 = 0.0314 (3.14%)
        expected_score = round(w_p_exp * 0.001 + w_l_exp * 0.08, 4)
        self.assertEqual(res["final_context_score"], expected_score)
        self.assertEqual(res["final_context_score_pct"], 3.14)

        mods = res["modalities"]
        self.assertFalse(mods["physiological"]["is_excluded"])
        self.assertFalse(mods["location"]["is_excluded"])
        self.assertTrue(mods["environment"]["is_excluded"])
        self.assertIn("offline or unavailable", mods["environment"]["exclusion_reason"])

    def test_06_all_modalities_excluded(self):
        """6. All modalities excluded: returns None score and INSUFFICIENT DATA classification (never 0 risk)."""
        res = self.engine.fuse(p_score=None, l_score=None, e_score=None)

        self.assertIn(res["status"], ["UNAVAILABLE", "INSUFFICIENT_DATA"])
        self.assertIsNone(res["final_context_score"])
        self.assertIsNone(res["final_context_score_pct"])
        self.assertIsNone(res["overall_context_index"])
        self.assertEqual(res["context_classification"], "INSUFFICIENT DATA")
        self.assertEqual(res["fusion_mode"], "INSUFFICIENT_DATA")

        # All modalities marked excluded
        mods = res["modalities"]
        self.assertTrue(mods["physiological"]["is_excluded"])
        self.assertTrue(mods["location"]["is_excluded"])
        self.assertTrue(mods["environment"]["is_excluded"])

        # Predictor returns INSUFFICIENT DATA
        pred_res = self.predictor.predict_fusion(
            physiological_stress_score=None,
            location_risk_score=None,
            environment_risk_score=None
        )
        self.assertIn(pred_res["status"], ["UNAVAILABLE", "INSUFFICIENT_DATA"])
        self.assertIsNone(pred_res["final_context_score"])
        self.assertIsNone(pred_res["threat_probability"])
        self.assertEqual(pred_res["context_classification"], "INSUFFICIENT DATA")

    def test_07_stale_modality_input(self):
        """7. Stale modality input: weight decays exponentially and drops to 0 when past staleness boundary."""
        tau_l = FRESHNESS_TAUS_SEC["spatial"]  # 300s
        res_fresh = self.engine.fuse(p_score=0.30, l_score=0.70, e_score=0.40, spatial_age_sec=0.0)
        res_stale = self.engine.fuse(p_score=0.30, l_score=0.70, e_score=0.40, spatial_age_sec=tau_l * 4.0)

        w_l_fresh = res_fresh["adaptive_weights"]["location"]
        w_l_stale = res_stale["adaptive_weights"]["location"]

        self.assertLess(w_l_stale, w_l_fresh)
        self.assertLess(w_l_stale, 0.02)  # exp(-4) ~ 0.0183 -> weight drops below 2%
        self.assertGreater(res_stale["adaptive_weights"]["physiological"], res_fresh["adaptive_weights"]["physiological"])

        # Weights still sum to 1.0
        w_sum = sum(res_stale["adaptive_weights"][m] for m in ("physiological", "location", "environment"))
        self.assertAlmostEqual(w_sum, 1.0, places=5)

    def test_08_invalid_modality_input(self):
        """8. Invalid modality input: NaN / Inf / out-of-range receives 0 weight and remaining modalities re-normalize."""
        # NaN
        res_nan = self.engine.fuse(p_score=float("nan"), l_score=0.50, e_score=0.30)
        self.assertEqual(res_nan["adaptive_weights"]["physiological"], 0.0)
        self.assertTrue(res_nan["modalities"]["physiological"]["is_excluded"])
        self.assertIn("Invalid", res_nan["modalities"]["physiological"]["exclusion_reason"])
        self.assertAlmostEqual(res_nan["adaptive_weights"]["location"] + res_nan["adaptive_weights"]["environment"], 1.0, places=5)

        # Inf
        res_inf = self.engine.fuse(p_score=0.20, l_score=float("inf"), e_score=0.30)
        self.assertEqual(res_inf["adaptive_weights"]["location"], 0.0)
        self.assertTrue(res_inf["modalities"]["location"]["is_excluded"])

        # Out-of-range (< 0 or > 1)
        res_out = self.engine.fuse(p_score=-0.5, l_score=0.50, e_score=1.5)
        self.assertEqual(res_out["adaptive_weights"]["physiological"], 0.0)
        self.assertEqual(res_out["adaptive_weights"]["environment"], 0.0)
        self.assertEqual(res_out["adaptive_weights"]["location"], 1.0)
        self.assertEqual(res_out["final_context_score"], 0.50)

    def test_09_zero_reliability_or_zero_total_weight(self):
        """9. Zero reliability or zero total effective weight: handled safely without ZeroDivisionError."""
        # Simulate disconnected wearable, GPS unavailable, and environmental offline
        res = self.engine.fuse(
            p_score=0.20,
            l_score=0.50,
            e_score=0.30,
            physio_telemetry={"connection_status": "DISCONNECTED"},
            spatial_analysis={"status": "GPS_UNAVAILABLE"},
            environmental_res={"status": "OFFLINE"}
        )
        self.assertIn(res["status"], ["UNAVAILABLE", "INSUFFICIENT_DATA"])
        self.assertIsNone(res["final_context_score"])
        self.assertEqual(res["context_classification"], "INSUFFICIENT DATA")
        self.assertEqual(res["adaptive_weights"]["physiological"], 0.0)
        self.assertEqual(res["adaptive_weights"]["location"], 0.0)
        self.assertEqual(res["adaptive_weights"]["environment"], 0.0)

    def test_10_weight_normalization_and_contribution_reconciliation(self):
        """10. Weight normalization and contribution reconciliation: weights sum to 100%, contributions sum to score."""
        scenarios = [
            (0.10, 0.85, 1.00),
            (0.002, 0.85, 1.00),
            (0.40, 0.70, None),
            (None, 0.30, 0.90),
            (0.80, None, 0.50),
            (0.55, None, None),
            (None, 0.65, None),
            (None, None, 0.45)
        ]
        for p, l, e in scenarios:
            res = self.engine.fuse(p, l, e)
            self.assertEqual(res["status"], "SUCCESS")
            weights = res["adaptive_weights"]
            w_sum = weights["physiological"] + weights["location"] + weights["environment"]
            self.assertAlmostEqual(w_sum, 1.0, places=5, msg=f"Weights must sum to 1.0 for ({p}, {l}, {e})")

            contribs = res["modality_contributions"]
            c_sum = contribs["physiological"] + contribs["location"] + contribs["environment"]
            self.assertAlmostEqual(c_sum, res["final_context_score"], places=4, msg=f"Contributions must sum to final score for ({p}, {l}, {e})")

    def test_11_dashboard_values_match_authoritative_result(self):
        """11. Dashboard values match the authoritative fusion result: no UNKNOWN/EXCLUDED when modalities contribute."""
        # User scenario: P=0.002 (0.2%), L=0.85 (85.0%), E=1.00 (100.0%)
        pred_res = self.predictor.predict_fusion(
            physiological_stress_score=0.002,
            location_risk_score=0.85,
            environment_risk_score=1.00
        )
        self.assertEqual(pred_res["status"], "SUCCESS")
        self.assertAlmostEqual(pred_res["threat_score_pct"], 56.33, places=1)

        with patch("streamlit.html") as mock_st_html:
            render_overall_risk_panel(pred_res)
            rendered = mock_st_html.call_args[0][0]

            # All three modalities contributed: NONE should be rendered as UNKNOWN / EXCLUDED!
            self.assertNotIn("Physiological Stress (Excluded · 0.0% Weight)", rendered)
            self.assertNotIn("Location Context Risk (Excluded · 0.0% Weight)", rendered)
            self.assertNotIn("Environmental Context (Excluded · 0.0% Weight)", rendered)

            # Check that adaptive weights and contributions are rendered correctly
            self.assertIn("40.0% Adaptive Weight", rendered)
            self.assertIn("25.0% Adaptive Weight", rendered)
            self.assertIn("35.0% Adaptive Weight", rendered)
            self.assertIn("56.33%", rendered)

    def test_12_missing_inputs_never_produce_fabricated_context_score(self):
        """12. Missing inputs never produce a fabricated context score: missing biometrics and location return INSUFFICIENT DATA."""
        pred_res = self.predictor.predict_fusion(
            physiological_stress_score=None,
            location_risk_score=None,
            environment_risk_score=None
        )
        self.assertIsNone(pred_res["final_context_score"])
        self.assertIsNone(pred_res["threat_probability"])
        self.assertEqual(pred_res["context_classification"], "INSUFFICIENT DATA")

        with patch("streamlit.html") as mock_st_html:
            render_overall_risk_panel(pred_res)
            rendered = mock_st_html.call_args[0][0]

            # Hero shows INSUFFICIENT DATA
            self.assertIn("INSUFFICIENT DATA", rendered)
            # Must NOT display an elevated, moderate, or standard risk classification
            self.assertNotIn("STANDARD CONTEXT", rendered)
            self.assertNotIn("ELEVATED CONTEXT", rendered)
            self.assertNotIn("HIGH CONTEXT", rendered)
            # Modality exclusion reasons are shown
            self.assertIn("Missing biometric telemetry", rendered)
            self.assertIn("Missing GPS / landmark coordinate", rendered)
            self.assertIn("Environmental context offline or unavailable", rendered)

    def test_13_structured_fusion_result_contract(self):
        """13. Authoritative fusion result contains all required Section 2 & Section 4 contract keys."""
        pred_res = self.predictor.predict_fusion(
            physiological_stress_score=0.001,
            location_risk_score=0.08,
            environment_risk_score=None
        )

        required_keys = [
            "final_context_score",
            "fusion_mode",
            "modalities",
            "modality_scores",
            "modality_reliability",
            "modality_freshness",
            "adaptive_weights",
            "excluded_modalities",
            "decision_rationale",
            "meta_learner_status"
        ]
        for key in required_keys:
            self.assertIn(key, pred_res, f"Missing required Section 2/4 key: {key}")

        # Check modality detail structure
        mods = pred_res["modalities"]
        for m_name in ("physiological", "location", "environment"):
            self.assertIn(m_name, mods)
            mod_item = mods[m_name]
            for field in (
                "modality_name", "raw_score", "is_available", "is_valid",
                "validity", "freshness", "source_reliability", "baseline_prior",
                "effective_weight", "normalized_weight", "weighted_contribution",
                "is_excluded", "exclusion_reason"
            ):
                self.assertIn(field, mod_item, f"Missing field '{field}' in modality '{m_name}'")

    def test_14_decision_rationale_dynamically_generated(self):
        """14. Decision rationale dynamically reports actual scores, weights, excluded modalities, and final score."""
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

    def test_15_dashboard_renders_single_panel_without_duplicate(self):
        """15. Dashboard render_overall_risk_panel renders exactly once (no duplicate st.html)."""
        pred_res = self.predictor.predict_fusion(
            physiological_stress_score=0.001,
            location_risk_score=0.08,
            environment_risk_score=None
        )
        with patch("streamlit.html") as mock_st_html:
            render_overall_risk_panel(pred_res)
            self.assertEqual(mock_st_html.call_count, 1, "render_overall_risk_panel must call st.html exactly once")

    def test_16_learned_gating_never_reported_as_trained_without_artifact(self):
        """16. Learned gating is transparently reported as NOT TRAINED without a valid empirical artifact."""
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
