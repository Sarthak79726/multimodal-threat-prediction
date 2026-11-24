"""
Unit and Regression Tests for Environmental Engine IST Time Handling & Fault Tolerance.
Covers:
1. Centralized IST time retrieval structure and keys.
2. Timezone fallback when ZoneInfo or IANA database is unavailable.
3. Midnight boundary and date transitions.
4. Predictor interface delegation and fault tolerance when env_engine is None or outdated.
5. Exact regression test for the Streamlit Cloud AttributeError (cached/outdated engine).
6. Missing model artifacts handling.
7. Unknown location handling.
"""

import sys
import unittest
import datetime
from pathlib import Path
from unittest.mock import patch, MagicMock

# Ensure project root is in sys.path
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import config
from src.environment.model import EnvironmentalContextEngine, SUPPORTED_LOCATIONS
from src.fusion.predict import MultimodalThreatPredictor

class TestEnvironmentalTimeAndFaultTolerance(unittest.TestCase):
    """Regression test suite for environmental time detection and fault tolerance."""

    def test_config_get_current_ist_time_structure(self):
        """Verifies that config.get_current_ist_time() returns a complete, standardized dictionary."""
        time_info = config.get_current_ist_time()
        self.assertIsInstance(time_info, dict)
        
        required_keys = [
            "datetime", "iso_string", "time_str", "date_str",
            "hour_of_day", "hour_float", "day_of_week", "day_of_week_num", "is_weekend"
        ]
        for key in required_keys:
            self.assertIn(key, time_info, f"Missing key in time_info: {key}")

        self.assertIsInstance(time_info["datetime"], datetime.datetime)
        self.assertIsInstance(time_info["iso_string"], str)
        self.assertIsInstance(time_info["time_str"], str)
        self.assertTrue(time_info["time_str"].endswith("IST"))
        self.assertIsInstance(time_info["date_str"], str)
        self.assertIsInstance(time_info["hour_of_day"], int)
        self.assertTrue(0 <= time_info["hour_of_day"] <= 23)
        self.assertIsInstance(time_info["hour_float"], float)
        self.assertTrue(0.0 <= time_info["hour_float"] < 24.0)
        self.assertIsInstance(time_info["day_of_week"], str)
        self.assertIn(time_info["day_of_week"], [
            "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"
        ])
        self.assertIsInstance(time_info["day_of_week_num"], int)
        self.assertTrue(0 <= time_info["day_of_week_num"] <= 6)
        self.assertIn(time_info["is_weekend"], [0, 1])

    def test_timezone_fallback_when_campus_timezone_fails(self):
        """Verifies safe fallback to fixed-offset UTC+05:30 if CAMPUS_TIMEZONE raises or is invalid."""
        with patch.object(datetime.datetime, "now") as mock_now:
            # First call raises (simulating zoneinfo issue), second call succeeds with fallback
            fixed_dt = datetime.datetime(2026, 11, 24, 15, 30, 0, tzinfo=datetime.timezone(datetime.timedelta(hours=5, minutes=30)))
            mock_now.side_effect = [Exception("ZoneInfo database missing"), fixed_dt]
            
            res = config.get_campus_now()
            self.assertEqual(res, fixed_dt)

    def test_midnight_boundary_and_date_transition(self):
        """Verifies correct calculation across midnight boundary and weekday/weekend transitions."""
        # 1. 23:59:59 on a Friday night
        dt_friday_night = datetime.datetime(2026, 11, 20, 23, 59, 59, tzinfo=config.CAMPUS_TIMEZONE)
        fri_res = config.format_ist_time_dict(dt_friday_night)
        self.assertEqual(fri_res["hour_of_day"], 23)
        self.assertAlmostEqual(fri_res["hour_float"], 23.9997, places=3)
        self.assertEqual(fri_res["day_of_week"], "Friday")
        self.assertEqual(fri_res["is_weekend"], 0)

        # 2. 00:00:00 on Saturday morning (midnight transition)
        dt_sat_midnight = datetime.datetime(2026, 11, 21, 0, 0, 0, tzinfo=config.CAMPUS_TIMEZONE)
        sat_res = config.format_ist_time_dict(dt_sat_midnight)
        self.assertEqual(sat_res["hour_of_day"], 0)
        self.assertEqual(sat_res["hour_float"], 0.0)
        self.assertEqual(sat_res["day_of_week"], "Saturday")
        self.assertEqual(sat_res["is_weekend"], 1)

        # 3. 23:59:59 on Sunday night
        dt_sun_night = datetime.datetime(2026, 11, 22, 23, 59, 59, tzinfo=config.CAMPUS_TIMEZONE)
        sun_res = config.format_ist_time_dict(dt_sun_night)
        self.assertEqual(sun_res["day_of_week"], "Sunday")
        self.assertEqual(sun_res["is_weekend"], 1)

        # 4. 00:00:00 on Monday morning
        dt_mon_midnight = datetime.datetime(2026, 11, 23, 0, 0, 0, tzinfo=config.CAMPUS_TIMEZONE)
        mon_res = config.format_ist_time_dict(dt_mon_midnight)
        self.assertEqual(mon_res["day_of_week"], "Monday")
        self.assertEqual(mon_res["is_weekend"], 0)

    def test_environmental_engine_get_current_ist_time(self):
        """Verifies that EnvironmentalContextEngine.get_current_ist_time() delegates to config."""
        res_engine = EnvironmentalContextEngine.get_current_ist_time()
        self.assertIn("time_str", res_engine)
        self.assertIn("day_of_week", res_engine)
        self.assertIn("hour_float", res_engine)

    def test_predictor_get_current_ist_time_public_interface(self):
        """Verifies that MultimodalThreatPredictor exposes a robust get_current_ist_time() method."""
        predictor = MultimodalThreatPredictor()
        res = predictor.get_current_ist_time()
        self.assertIn("time_str", res)
        self.assertIn("day_of_week", res)

    def test_predictor_resilience_when_env_engine_is_none(self):
        """Verifies that MultimodalThreatPredictor operates safely even if self.env_engine is None."""
        predictor = MultimodalThreatPredictor()
        predictor.env_engine = None  # Simulate engine failure

        # Time retrieval should still work via fallback
        time_info = predictor.get_current_ist_time()
        self.assertIn("time_str", time_info)

        # predict_environment score fallback
        score = predictor.predict_environment(14.0, "Light", "Medium")
        self.assertEqual(score, 0.20)

        # predict_environment_context fallback
        context_res = predictor.predict_environment_context(location="Main Building", hour=14.0)
        self.assertEqual(context_res["status"], "UNAVAILABLE")
        self.assertEqual(context_res["crowd_prediction"]["category"], "UNKNOWN")
        self.assertEqual(context_res["lighting_prediction"]["category"], "UNKNOWN")
        self.assertEqual(context_res["calibrated_environmental_score"], 0.20)

        # predict_fusion fallback
        fusion_res = predictor.predict_fusion(0.20, 0.20, 0.20)
        self.assertIn("threat_probability", fusion_res)
        self.assertIn("environmental", fusion_res["modalities_metadata"])
        self.assertEqual(fusion_res["modalities_metadata"]["environmental"]["status"], "UNAVAILABLE")

    def test_streamlit_cloud_cached_attribute_error_regression(self):
        """
        Exact regression test for the Streamlit Cloud crash:
        Simulates an outdated/cached predictor whose env_engine lacks get_current_ist_time.
        Tests both predictor.get_current_ist_time() and the app/main.py defensive block.
        """
        # Create a mock outdated env_engine with NO get_current_ist_time attribute
        class OutdatedEnvEngine:
            pass

        predictor = MultimodalThreatPredictor()
        predictor.env_engine = OutdatedEnvEngine()  # Missing get_current_ist_time attribute

        # Predictor's own delegate should handle this gracefully
        time_info = predictor.get_current_ist_time()
        self.assertIn("time_str", time_info)

        # Simulate the exact defensive logic in app/main.py (lines 507-515)
        try:
            if hasattr(predictor, "get_current_ist_time"):
                app_time_info = predictor.get_current_ist_time()
            elif hasattr(predictor, "env_engine") and predictor.env_engine is not None and hasattr(predictor.env_engine, "get_current_ist_time"):
                app_time_info = predictor.env_engine.get_current_ist_time()
            else:
                app_time_info = config.get_current_ist_time()
        except Exception:
            app_time_info = config.get_current_ist_time()

        self.assertIn("time_str", app_time_info)
        self.assertIn("day_of_week", app_time_info)
        self.assertIn("hour_float", app_time_info)

    def test_engine_unknown_location_safe_handling(self):
        """Verifies that an unsupported or out-of-campus location produces UNKNOWN status without crashing."""
        engine = EnvironmentalContextEngine()
        res = engine.predict_environment(location="Invalid Remote Location 123", hour_of_day=14.0)
        self.assertEqual(res["status"], "UNKNOWN")
        self.assertEqual(res["error"], "UNSUPPORTED_LOCATION")
        self.assertEqual(res["crowd_prediction"]["category"], "UNKNOWN")
        self.assertEqual(res["lighting_prediction"]["category"], "UNKNOWN")
        self.assertEqual(res["calibrated_environmental_score"], 0.20)

    def test_engine_missing_artifacts_safe_handling(self):
        """Verifies that missing model files result in UNKNOWN status and do not raise unhandled exceptions."""
        engine = EnvironmentalContextEngine(
            crowd_model_path=Path("nonexistent_crowd.pkl"),
            lighting_model_path=Path("nonexistent_lighting.pkl")
        )
        self.assertFalse(engine.is_trained_ml)
        self.assertIsNotNone(engine.load_error)

        res = engine.predict_environment(location="Main Building", hour_of_day=14.0)
        self.assertEqual(res["status"], "UNKNOWN")
        self.assertEqual(res["error"], "MODEL_UNAVAILABLE")
        self.assertEqual(res["crowd_prediction"]["category"], "UNKNOWN")
        self.assertEqual(res["lighting_prediction"]["category"], "UNKNOWN")

if __name__ == "__main__":
    unittest.main()
