"""
Production Deployment Configuration and Verification Test Suite.
Verifies that all 3 modalities and the adaptive fusion engine execute reliably
under production cloud deployment constraints (Streamlit Cloud, Linux/Docker, headless mode).

Tests:
1. Physiological inference (WESAD RF artifact, 76-feature schema, non-raw-pickle fallback).
2. Location inference (Campus landmarks, Haversine distance, geofencing, score generation).
3. Environmental inference (Open-Meteo weather client, schedule models, composite score).
4. Full three-modality fusion (Reliability-aware adaptive weights 40/25/35).
5. Partial fusion (Resilient dual-branch re-normalization when weather/env is offline).
6. Missing artifact handling (Graceful degradation when artifacts are missing).
7. API failure handling (Open-Meteo network timeout / HTTP error resilience).
"""

import sys
import unittest
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

# Ensure project root is prioritized
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import config
from src.fusion.predict import MultimodalThreatPredictor
from src.location.campus_map import CampusMapAnalyzer
from src.ingestion.weather_client import OpenMeteoWeatherClient
from app.main import evaluate_subject_biometrics, get_deployment_commit_info


class TestProductionDeployment(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.predictor = MultimodalThreatPredictor()
        cls.campus_analyzer = CampusMapAnalyzer()

    def test_01_physiological_production_inference(self):
        """1. Physiological inference loads 76-feature artifact and computes valid stress score."""
        p_art = config.MODELS_DIR / "physiological" / "physiological_model.pkl"
        self.assertTrue(p_art.exists(), f"Artifact missing at {p_art}")
        self.assertTrue(self.predictor.physio_engine.is_trained_ml, "WESAD RF model must be loaded")
        self.assertEqual(len(self.predictor.physio_engine.feature_names), 76, "Feature count must be exactly 76")

        # Test inference via repository condition fallback (production environment without 20GB raw pickle)
        with patch.object(config, "WESAD_DIR", Path("/non_existent_wesad_dir")):
            res = evaluate_subject_biometrics("S2", "Window 1 (Baseline)")
            self.assertIsNotNone(res)
            self.assertIn("physio_score", res)
            self.assertTrue(0.0 <= res["physio_score"] <= 1.0)
            self.assertEqual(res["predicted_class"], "NON-STRESS")
            self.assertEqual(res["subject_id"], "S2")
            self.assertIn("ecg_waveform_preview", res)
            self.assertGreater(len(res["ecg_waveform_preview"]), 0)

    def test_02_location_production_inference(self):
        """2. Location inference resolves campus coordinates and generates spatial score."""
        self.assertGreaterEqual(len(config.BIT_MESRA_LANDMARKS), 17, "Must contain all 17 campus landmarks")
        target_lm = config.BIT_MESRA_LANDMARKS[0]
        loc_res = self.campus_analyzer.analyze_gps(
            lat=float(target_lm["latitude"]),
            lon=float(target_lm["longitude"]),
            accuracy=5.0,
            time_of_day_hour=14.0
        )
        self.assertIsNotNone(loc_res)
        self.assertIn("location_risk_score", loc_res)
        self.assertTrue(0.0 <= loc_res["location_risk_score"] <= 1.0)
        self.assertIn("distance_to_security_m", loc_res)
        self.assertEqual(loc_res["nearest_landmark"]["name"], target_lm["name"])

    def test_03_environmental_production_inference(self):
        """3. Environmental inference retrieves weather reading and generates composite score."""
        res = self.predictor.predict_environment_context(
            location="Main Building",
            hour=14.0,
            day_of_week="Thursday",
            is_weekend=0
        )
        self.assertIsNotNone(res)
        self.assertEqual(res["status"], "SUCCESS")
        self.assertIn("environmental_context_score", res)
        self.assertTrue(0.0 <= res["environmental_context_score"] <= 1.0)
        self.assertIn("weather_reading", res)

    def test_04_full_three_modality_production_fusion(self):
        """4. Full three-modality fusion calculates nominal baseline priors (40/25/35)."""
        fusion_res = self.predictor.predict_fusion(
            physiological_stress_score=0.10,
            location_risk_score=0.20,
            environment_risk_score=0.30,
            physio_age_sec=0.0,
            spatial_age_sec=0.0,
            env_age_sec=0.0,
            is_manual_override=True
        )
        self.assertEqual(fusion_res["status"], "SUCCESS")
        self.assertEqual(fusion_res["fusion_mode"], "THREE_BRANCH_FUSION")
        self.assertIsNotNone(fusion_res["final_context_score"])
        self.assertAlmostEqual(fusion_res["final_context_score"], 0.195, places=3)
        weights = fusion_res["adaptive_weights"]
        self.assertAlmostEqual(weights["physiological"], 0.40, places=2)
        self.assertAlmostEqual(weights["location"], 0.25, places=2)
        self.assertAlmostEqual(weights["environment"], 0.35, places=2)

    def test_05_partial_fusion_production_handling(self):
        """5. Partial dual-branch fusion re-normalizes weights when environmental is offline."""
        fusion_res = self.predictor.predict_fusion(
            physiological_stress_score=0.10,
            location_risk_score=0.20,
            environment_risk_score=None,
            physio_age_sec=0.0,
            spatial_age_sec=0.0,
            is_manual_override=True
        )
        self.assertEqual(fusion_res["status"], "SUCCESS")
        self.assertEqual(fusion_res["fusion_mode"], "PARTIAL_DUAL_BRANCH")
        # Re-normalized weights: 0.40/(0.40+0.25) = 61.54%, 0.25/(0.40+0.25) = 38.46%
        weights = fusion_res["adaptive_weights"]
        self.assertAlmostEqual(weights["physiological"], 0.6154, places=3)
        self.assertAlmostEqual(weights["location"], 0.3846, places=3)
        self.assertEqual(weights["environment"], 0.0)

    def test_06_missing_artifact_graceful_handling(self):
        """6. Missing model artifacts degrade gracefully to safe baseline without unhandled crashes."""
        dummy_predictor = MultimodalThreatPredictor()
        # Simulate missing physiological model
        dummy_predictor.physio_engine.is_trained_ml = False
        dummy_predictor.physio_engine.model = None
        fallback_score = dummy_predictor.predict_physiological({})
        self.assertTrue(0.0 <= fallback_score <= 1.0)
        self.assertEqual(fallback_score, 0.20)

    def test_07_api_failure_handling(self):
        """7. Open-Meteo API network failure degrades gracefully to diurnal context."""
        with patch("urllib.request.urlopen", side_effect=Exception("Connection timed out")):
            client = OpenMeteoWeatherClient()
            client._cache = None
            client._last_fetch_time = None
            reading = client.get_current_weather(force_refresh=True)
            self.assertIsNotNone(reading)
            self.assertEqual(reading.status.value, "UNAVAILABLE")

        # Environmental engine prediction handles unavailable weather gracefully
        res = self.predictor.predict_environment_context(
            location="Main Building",
            hour=14.0,
            day_of_week="Thursday",
            is_weekend=0
        )
        self.assertEqual(res["status"], "SUCCESS")
        self.assertIsNotNone(res["environmental_context_score"])


if __name__ == "__main__":
    unittest.main()
