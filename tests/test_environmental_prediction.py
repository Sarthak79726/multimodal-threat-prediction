"""
Comprehensive Unit & Reliability Tests for Environmental Context AI Models.
Tests:
- All 17 supported BIT Mesra campus locations
- All 24 hours (diurnal coverage)
- Weekday vs Weekend sensitivity
- Midnight and day transitions
- Unknown or out-of-bounds locations (rejection / UNKNOWN status)
- GPS resolution and geofencing threshold
- GPS unavailable fallback to manual selection
- Missing model artifacts & load failure resilience
- Invalid input features handling
- Prediction schema and synthetic data provenance tags
- Environmental-to-Multimodal fusion integration
"""

import unittest
from pathlib import Path
import numpy as np

import config
from src.environment.model import EnvironmentalContextEngine, SUPPORTED_LOCATIONS
from src.fusion.predict import MultimodalThreatPredictor

class TestEnvironmentalPrediction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = EnvironmentalContextEngine()
        cls.predictor = MultimodalThreatPredictor()

    def test_engine_is_trained_ml(self):
        """Verifies that trained environmental model artifacts are successfully loaded."""
        self.assertTrue(self.engine.is_trained_ml)
        self.assertIsNotNone(self.engine.crowd_artifact)
        self.assertIsNotNone(self.engine.lighting_artifact)

    def test_all_17_campus_locations(self):
        """Verifies that all 17 BIT Mesra campus landmarks produce valid predictions."""
        self.assertEqual(len(SUPPORTED_LOCATIONS), 17)
        for loc in SUPPORTED_LOCATIONS:
            res = self.engine.predict_environment(
                location=loc,
                hour_of_day=14.0,
                day_of_week="Wednesday",
                weekend_indicator=0
            )
            self.assertEqual(res["status"], "SUCCESS", f"Failed for location: {loc}")
            self.assertIn(res["crowd_prediction"]["category"], ["Low", "Medium", "High"])
            self.assertIn(res["lighting_prediction"]["category"], ["Dark", "Low", "Good"])
            self.assertGreaterEqual(res["calibrated_environmental_score"], 0.05)
            self.assertLessEqual(res["calibrated_environmental_score"], 1.0)
            self.assertEqual(res["synthetic_provenance"], "SYNTHETIC_SIMULATION_RULE_GENERATED")

    def test_all_24_hours(self):
        """Verifies prediction across all 24 diurnal hours."""
        for hour in range(24):
            res = self.engine.predict_environment(
                location="Main Building",
                hour_of_day=float(hour),
                day_of_week="Monday",
                weekend_indicator=0
            )
            self.assertEqual(res["status"], "SUCCESS")
            self.assertIn(res["lighting_prediction"]["category"], ["Dark", "Low", "Good"])
            self.assertIn(res["crowd_prediction"]["category"], ["Low", "Medium", "High"])

    def test_weekday_vs_weekend_academic_reduction(self):
        """Verifies academic areas (Main Building) experience crowd reduction on weekends."""
        weekday_res = self.engine.predict_environment(
            location="Main Building",
            hour_of_day=11.0,
            day_of_week="Tuesday",
            weekend_indicator=0
        )
        weekend_res = self.engine.predict_environment(
            location="Main Building",
            hour_of_day=11.0,
            day_of_week="Sunday",
            weekend_indicator=1
        )
        # Weekday daytime crowd in Main Building should be High
        self.assertEqual(weekday_res["crowd_prediction"]["category"], "High")
        # Weekend crowd probability for High should be significantly lower than on weekday
        p_high_weekday = weekday_res["crowd_prediction"]["probabilities"].get("High", 0.0)
        p_high_weekend = weekend_res["crowd_prediction"]["probabilities"].get("High", 0.0)
        self.assertGreater(p_high_weekday, p_high_weekend)

    def test_midnight_boundary(self):
        """Verifies midnight transitions (23.99 to 0.01) execute smoothly."""
        res_pre = self.engine.predict_environment(
            location="H10",
            hour_of_day=23.99,
            day_of_week="Friday",
            weekend_indicator=0
        )
        res_post = self.engine.predict_environment(
            location="H10",
            hour_of_day=0.01,
            day_of_week="Saturday",
            weekend_indicator=1
        )
        self.assertEqual(res_pre["status"], "SUCCESS")
        self.assertEqual(res_post["status"], "SUCCESS")
        self.assertIn(res_pre["lighting_prediction"]["category"], ["Dark", "Low"])
        self.assertIn(res_post["lighting_prediction"]["category"], ["Dark", "Low"])

    def test_unknown_location_returns_unknown(self):
        """Verifies that unrecognized locations return UNKNOWN status rather than fabricating predictions."""
        res = self.engine.predict_environment(
            location="NonExistentBuilding",
            hour_of_day=14.0,
            day_of_week="Monday",
            weekend_indicator=0
        )
        self.assertEqual(res["status"], "UNKNOWN")
        self.assertEqual(res["crowd_prediction"]["category"], "UNKNOWN")
        self.assertEqual(res["lighting_prediction"]["category"], "UNKNOWN")
        self.assertIn("UNSUPPORTED_LOCATION", res.get("error", ""))

    def test_gps_resolution_inside_geofence(self):
        """Verifies accurate GPS coordinate resolution for a known campus landmark."""
        h10_coords = config.BIT_MESRA_LANDMARKS_DB["h10"]
        gps_res = self.engine.resolve_campus_location(
            lat=h10_coords["latitude"],
            lon=h10_coords["longitude"],
            max_geofence_meters=200.0
        )
        self.assertEqual(gps_res["status"], "RESOLVED_GPS")
        self.assertEqual(gps_res["resolved_location"], "H10")
        self.assertTrue(gps_res["is_within_geofence"])
        self.assertLessEqual(gps_res["distance_meters"], 5.0)

    def test_gps_resolution_outside_geofence(self):
        """Verifies coordinates far outside campus are rejected and not silently snapped."""
        # Coordinates in central Ranchi city (~15 km away)
        gps_res = self.engine.resolve_campus_location(
            lat=23.3441,
            lon=85.3096,
            max_geofence_meters=350.0
        )
        self.assertEqual(gps_res["status"], "OUTSIDE_CAMPUS_GEOFENCE")
        self.assertIsNone(gps_res["resolved_location"])
        self.assertFalse(gps_res["is_within_geofence"])

    def test_gps_unavailable_fallback(self):
        """Verifies that missing GPS coordinates return GPS_UNAVAILABLE requiring manual selection."""
        gps_res = self.engine.resolve_campus_location(lat=None, lon=None)
        self.assertEqual(gps_res["status"], "GPS_UNAVAILABLE")
        self.assertIsNone(gps_res["resolved_location"])
        self.assertFalse(gps_res["is_within_geofence"])

    def test_missing_model_artifact_resilience(self):
        """Verifies graceful UNKNOWN handling when artifact paths do not exist."""
        dummy_engine = EnvironmentalContextEngine(
            crowd_model_path=Path("non_existent_crowd.pkl"),
            lighting_model_path=Path("non_existent_lighting.pkl")
        )
        self.assertFalse(dummy_engine.is_trained_ml)
        res = dummy_engine.predict_environment(location="Main Building", hour_of_day=12.0)
        self.assertEqual(res["status"], "UNKNOWN")
        self.assertEqual(res["crowd_prediction"]["category"], "UNKNOWN")

    def test_invalid_feature_input_handling(self):
        """Verifies handling of extreme, negative, or wrapped hour values."""
        res_wrap = self.engine.predict_environment(
            location="Main Gate",
            hour_of_day=26.5,
            day_of_week="Monday",
            weekend_indicator=0
        )
        self.assertEqual(res_wrap["status"], "SUCCESS")

    def test_prediction_output_schema_and_provenance(self):
        """Verifies complete standardized schema and explicit synthetic provenance disclosure."""
        res = self.engine.predict_environment(
            location="H10",
            hour_of_day=22.5,
            day_of_week="Tuesday",
            weekend_indicator=0
        )
        self.assertIn("status", res)
        self.assertIn("location", res)
        self.assertIn("time_display", res)
        self.assertIn("crowd_prediction", res)
        self.assertIn("lighting_prediction", res)
        self.assertIn("calibrated_environmental_score", res)
        self.assertIn("environmental_crowd_context_score", res)
        self.assertIn("environmental_lighting_context_score", res)
        self.assertIn("synthetic_provenance", res)
        self.assertIn("warning", res)

        # Check crowd prediction schema
        cp = res["crowd_prediction"]
        self.assertIn("category", cp)
        self.assertIn("confidence", cp)
        self.assertIn("model_name", cp)
        self.assertIn("provenance", cp)
        self.assertGreaterEqual(cp["confidence"], 0.0)
        self.assertLessEqual(cp["confidence"], 1.0)

        # Check lighting prediction schema
        lp = res["lighting_prediction"]
        self.assertIn("category", lp)
        self.assertIn("confidence", lp)
        self.assertIn("lux_estimate", lp)
        self.assertIn("model_name", lp)
        self.assertIn("provenance", lp)

    def test_environmental_to_fusion_integration(self):
        """Verifies that MultimodalThreatPredictor integrates the environmental branch."""
        env_res = self.predictor.predict_environment_context(location="H13", hour=23.0)
        self.assertEqual(env_res["status"], "SUCCESS")
        
        env_score = env_res["calibrated_environmental_score"]
        fusion_res = self.predictor.predict_fusion(
            physiological_stress_score=0.20,
            location_risk_score=0.85,
            environment_risk_score=env_score
        )
        self.assertIn("environmental_crowd_context_score", fusion_res)
        self.assertIn("environmental_lighting_context_score", fusion_res)
        self.assertIn("environmental_model_provenance", fusion_res)
        self.assertEqual(fusion_res["environment_risk_score"], env_score)
        self.assertFalse(fusion_res["is_meta_learner_trained"])
        self.assertEqual(fusion_res["meta_model_status"], "UNTRAINED — VALID MULTIMODAL THREAT DATASET REQUIRED")

if __name__ == "__main__":
    unittest.main()
