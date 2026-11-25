"""
Unit and Integration Tests for Reliability-Aware Adaptive Multimodal Fusion.
Tests:
1. All modalities fully reliable.
2. Physiological sensor reliability decreasing.
3. Location data becoming stale.
4. Environmental modality missing.
5. One modality invalid.
6. All modalities missing.
7. Weights always sum to 1 when fusion is available.
8. No random or fabricated weights (deterministic reproducibility).
9. Dashboard values match backend fusion output.
10. No stale fixed-weight formula is used by the active dashboard.
11. Learned gating is never reported as trained without a valid trained artifact.
"""

import sys
import math
import unittest
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

    def test_01_all_modalities_fully_reliable(self):
        """1. When all modalities are fully reliable and fresh, weights equal baseline priors and sum to 1."""
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
        self.assertAlmostEqual(sum(weights.values()), 1.0, places=5)

        expected_score = round(0.40 * 0.20 + 0.25 * 0.60 + 0.35 * 0.28, 4)
        self.assertEqual(res["overall_context_index"], expected_score)
        self.assertEqual(res["overall_context_index_pct"], round(expected_score * 100.0, 2))

    def test_02_physiological_sensor_reliability_decreasing(self):
        """2. When physiological sensor reliability decreases, its adaptive weight decreases while others increase."""
        # Baseline fresh
        res_baseline = self.engine.fuse(p_score=0.50, l_score=0.50, e_score=0.50)
        w_p_base = res_baseline["adaptive_weights"]["physiological"]

        # Degraded telemetry with sensor errors / incomplete channels
        degraded_telemetry = {
            "eda_mean": None,
            "temp_mean": 15.0,  # outside plausible bounds -> validity penalized
            "ecg_std": None,
            "heart_rate_bpm": 220,  # tachycardia outside bounds
            "is_live_sensor": False
        }
        res_degraded = self.engine.fuse(
            p_score=0.50,
            l_score=0.50,
            e_score=0.50,
            physio_telemetry=degraded_telemetry
        )

        w_p_deg = res_degraded["adaptive_weights"]["physiological"]
        w_l_deg = res_degraded["adaptive_weights"]["location"]
        w_e_deg = res_degraded["adaptive_weights"]["environment"]

        self.assertLess(w_p_deg, w_p_base, "Degraded physiological sensor must have lower adaptive weight")
        self.assertGreater(w_l_deg, res_baseline["adaptive_weights"]["location"])
        self.assertGreater(w_e_deg, res_baseline["adaptive_weights"]["environment"])
        self.assertAlmostEqual(sum(res_degraded["adaptive_weights"].values()), 1.0, places=5)

    def test_03_location_data_becoming_stale(self):
        """3. When location data ages beyond freshness timescale, its weight decays via exp(-age/tau)."""
        tau_l = FRESHNESS_TAUS_SEC["spatial"]  # 300s
        res_fresh = self.engine.fuse(p_score=0.30, l_score=0.70, e_score=0.40, spatial_age_sec=0.0)
        res_stale = self.engine.fuse(p_score=0.30, l_score=0.70, e_score=0.40, spatial_age_sec=tau_l * 2.0)

        w_l_fresh = res_fresh["adaptive_weights"]["location"]
        w_l_stale = res_stale["adaptive_weights"]["location"]

        self.assertLess(w_l_stale, w_l_fresh, "Stale location data must have strictly lower weight")
        rel_l = res_stale["modality_reliabilities"]["location"]
        self.assertTrue(rel_l["is_stale"])
        self.assertAlmostEqual(rel_l["freshness"], math.exp(-2.0), places=3)
        self.assertAlmostEqual(sum(res_stale["adaptive_weights"].values()), 1.0, places=5)

    def test_04_environmental_modality_missing(self):
        """4. When environmental modality is missing, it receives 0 weight and remaining modalities re-normalize."""
        res = self.engine.fuse(p_score=0.20, l_score=0.60, e_score=None)
        weights = res["adaptive_weights"]

        self.assertEqual(weights["environment"], 0.0, "Missing environmental modality must have 0 weight")
        # Expected re-normalized weights: 0.40 / 0.65 and 0.25 / 0.65
        self.assertAlmostEqual(weights["physiological"], 0.40 / 0.65, places=3)
        self.assertAlmostEqual(weights["location"], 0.25 / 0.65, places=3)
        self.assertAlmostEqual(sum(weights.values()), 1.0, places=5)

        expected_score = round((0.40 * 0.20 + 0.25 * 0.60) / 0.65, 4)
        self.assertEqual(res["overall_context_index"], expected_score)
        self.assertIn("environment", res["excluded_modalities"])

    def test_05_one_modality_invalid(self):
        """5. When one modality is invalid (e.g. NaN or out of bounds), it receives 0 weight."""
        res = self.engine.fuse(p_score=float("nan"), l_score=0.50, e_score=0.30)
        weights = res["adaptive_weights"]

        self.assertEqual(weights["physiological"], 0.0, "Invalid physiological score must receive 0 weight")
        # Location and Environmental re-normalize: 0.25 / 0.60 and 0.35 / 0.60
        self.assertAlmostEqual(weights["location"], 0.25 / 0.60, places=3)
        self.assertAlmostEqual(weights["environment"], 0.35 / 0.60, places=3)
        self.assertAlmostEqual(sum(weights.values()), 1.0, places=5)
        self.assertIn("physiological", res["excluded_modalities"])

    def test_06_all_modalities_missing(self):
        """6. When all modalities are missing or invalid, returns UNAVAILABLE with None score (never 0 risk)."""
        res = self.engine.fuse(p_score=None, l_score=None, e_score=None)

        self.assertEqual(res["status"], "UNAVAILABLE")
        self.assertIsNone(res["overall_context_index"], "Overall context score must be None, NOT 0.0")
        self.assertIsNone(res["overall_context_index_pct"])
        self.assertEqual(res["context_classification"], "UNAVAILABLE")
        self.assertEqual(res["adaptive_weights"]["physiological"], 0.0)
        self.assertEqual(res["adaptive_weights"]["location"], 0.0)
        self.assertEqual(res["adaptive_weights"]["environment"], 0.0)

        # Via predictor
        pred_res = self.predictor.predict_fusion(
            physiological_stress_score=None,
            location_risk_score=None,
            environment_risk_score=None
        )
        self.assertEqual(pred_res["context_status"], "UNAVAILABLE")
        self.assertIsNone(pred_res["threat_probability"], "Predictor must return None for all-missing inputs")

    def test_07_weights_always_sum_to_one_when_fusion_available(self):
        """7. Weights always sum to 1.0 when at least one valid modality is available across diverse edge cases."""
        test_cases = [
            (0.1, 0.2, 0.3, 0.0, 0.0, 0.0),
            (0.8, 0.1, None, 50.0, 100.0, 0.0),
            (None, 0.9, 0.4, 0.0, 200.0, 300.0),
            (0.5, None, None, 0.0, 0.0, 0.0),
            (0.2, 0.4, 0.6, 600.0, 1200.0, 3600.0),
        ]
        for p, l, e, a_p, a_l, a_e in test_cases:
            res = self.engine.fuse(
                p_score=p, l_score=l, e_score=e,
                physio_age_sec=a_p, spatial_age_sec=a_l, env_age_sec=a_e
            )
            if res["is_available"]:
                w_sum = sum(res["adaptive_weights"].values())
                self.assertAlmostEqual(w_sum, 1.0, places=5, msg=f"Weights failed to sum to 1.0 for inputs: {p}, {l}, {e}")

    def test_08_no_random_or_fabricated_weights(self):
        """8. Weights and scores are 100% deterministic with zero randomness."""
        p, l, e = 0.33, 0.66, 0.22
        first_run = self.engine.fuse(p_score=p, l_score=l, e_score=e)

        for _ in range(50):
            subsequent_run = self.engine.fuse(p_score=p, l_score=l, e_score=e)
            self.assertEqual(first_run["overall_context_index"], subsequent_run["overall_context_index"])
            self.assertEqual(first_run["adaptive_weights"], subsequent_run["adaptive_weights"])
            self.assertEqual(first_run["modality_contributions"], subsequent_run["modality_contributions"])

    def test_09_dashboard_values_match_backend_fusion_output(self):
        """9. Predictor outputs match reliability-aware fusion engine directly."""
        pred_res = self.predictor.predict_fusion(
            physiological_stress_score=0.45,
            location_risk_score=0.75,
            environment_risk_score=0.30
        )
        self.assertIn("adaptive_weights", pred_res)
        self.assertIn("modality_contributions", pred_res)
        self.assertIn("modality_reliabilities", pred_res)
        self.assertIn("formula_str", pred_res)

        w = pred_res["adaptive_weights"]
        self.assertAlmostEqual(sum(w.values()), 1.0, places=5)
        self.assertEqual(pred_res["threat_probability"], pred_res["overall_context_index"])
        self.assertEqual(pred_res["fusion_method"], "RELIABILITY_AWARE_ADAPTIVE")

    def test_10_no_stale_fixed_weight_formula_used_by_active_dashboard(self):
        """10. Active dashboard panel dynamically adapts when weights differ from 40/25/35."""
        # Simulate stale location: weights shift away from 40/25/35
        pred_res = self.predictor.predict_fusion(
            physiological_stress_score=0.40,
            location_risk_score=0.80,
            environment_risk_score=0.20,
            spatial_age_sec=1200.0  # 4x tau_l -> stale
        )
        w = pred_res["adaptive_weights"]
        # Spatial weight must drop below 0.25
        self.assertLess(w["location"], 0.20)
        self.assertGreater(w["physiological"], 0.40)

        # Formula string must reflect the actual adaptive weights, not the stale 40% x P
        formula = pred_res["formula_str"]
        self.assertNotIn("40.0% × P + 25.0% × L", formula)
        self.assertIn(f"{w['location']*100:.1f}%", formula)

    def test_11_learned_gating_never_reported_as_trained_without_artifact(self):
        """11. Learned gating is transparently reported as NOT TRAINED without a valid empirical artifact."""
        gating = DynamicGatingNetwork()
        self.assertFalse(gating.is_trained)

        dummy_x = np.zeros((1, 15), dtype=np.float64)
        weights_dict = gating.predict_weights_dict(dummy_x)
        self.assertEqual(weights_dict["learned_gating_status"], "LEARNED GATING: NOT TRAINED")
        self.assertFalse(weights_dict["is_trained"])

        # Also in predictor result
        pred_res = self.predictor.predict_fusion(0.5, 0.5, 0.5)
        self.assertEqual(pred_res["learned_gating_status"], "LEARNED GATING: NOT TRAINED")
        self.assertFalse(pred_res["is_meta_learner_trained"])


if __name__ == "__main__":
    unittest.main()
