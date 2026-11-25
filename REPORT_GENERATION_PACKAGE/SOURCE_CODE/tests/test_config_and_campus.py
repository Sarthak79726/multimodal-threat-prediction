"""
Unit Tests for Configuration, Timezone, and Campus Geospatial Modules.
Tests:
- config.get_campus_now() returns datetime localized to Asia/Kolkata (IST: UTC+5:30).
- Landmark database integrity (17 verified BIT Mesra landmarks).
- Haversine distance metric calculation accuracy.
- CampusMapAnalyzer GPS geofencing.
"""

import unittest
import datetime
from pathlib import Path

import config
from src.location.campus_map import CampusMapAnalyzer, haversine_distance

class TestConfigAndCampus(unittest.TestCase):
    def test_campus_timezone_and_get_campus_now(self):
        """Verifies get_campus_now() does not raise AttributeError and produces valid localized IST time."""
        now = config.get_campus_now()
        self.assertIsInstance(now, datetime.datetime)
        self.assertIsNotNone(now.tzinfo)
        # Check offset is +5:30 (19800 seconds)
        offset = now.utcoffset()
        self.assertEqual(offset.total_seconds(), 5.5 * 3600)

    def test_landmarks_database_integrity(self):
        """Verifies that all 17 BIT Mesra landmarks are properly loaded with coordinates and required fields."""
        landmarks = config.BIT_MESRA_LANDMARKS
        self.assertGreaterEqual(len(landmarks), 17)

        required_keys = ["id", "name", "latitude", "longitude", "radius", "location_type", "lighting_level", "isolation_index"]
        for lm in landmarks:
            for k in required_keys:
                self.assertIn(k, lm, f"Landmark {lm.get('id')} missing required key '{k}'")
            # Verify latitude/longitude near Ranchi (lat ~23.4, lon ~85.4)
            self.assertAlmostEqual(lm["latitude"], 23.41, delta=0.05)
            self.assertAlmostEqual(lm["longitude"], 85.43, delta=0.05)

    def test_haversine_accuracy(self):
        """Verifies Haversine distance formula with known geographic distance."""
        # Main Building Clock Tower coordinates
        mb_lat, mb_lon = 23.412300, 85.439900
        # Main Gate coordinates
        gate_lat, gate_lon = 23.405500, 85.435000
        dist = haversine_distance(mb_lat, mb_lon, gate_lat, gate_lon)
        # Expected distance is ~910 meters
        self.assertGreater(dist, 850.0)
        self.assertLess(dist, 980.0)

        # Distance to itself should be 0.0
        self.assertAlmostEqual(haversine_distance(mb_lat, mb_lon, mb_lat, mb_lon), 0.0, places=5)

    def test_campus_map_analyzer(self):
        """Verifies CampusMapAnalyzer handles coordinate analysis and returns standardized schema."""
        analyzer = CampusMapAnalyzer()
        res = analyzer.analyze_gps(lat=23.4123, lon=85.4399, accuracy=10.0, time_of_day_hour=14.0)

        self.assertEqual(res["status"], "CONNECTED")
        self.assertIsNotNone(res["nearest_landmark"])
        self.assertEqual(res["nearest_landmark"]["id"], "main_building")
        self.assertTrue(res["detected_landmark"]["is_inside"])
        self.assertIsInstance(res["location_risk_score"], float)

        # Test None input handling
        res_none = analyzer.analyze_gps(lat=None, lon=None)
        self.assertEqual(res_none["status"], "ERROR")
        self.assertIsNone(res_none["current_location"])

if __name__ == "__main__":
    unittest.main()

