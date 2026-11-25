"""
Unit Tests for Base Model Prediction Interfaces and Metadata.
Tests:
- PhysiologicalStressModel prediction interface, output range [0, 1], feature dimension alignment, missing feature handling.
- SpatialRiskEngine prediction interface, deterministic Haversine calculations, output range [0, 1].
- EnvironmentalContextEngine prediction interface, time, lighting, crowd sensitivity, output range [0, 1].
"""

import unittest
import numpy as np
import pandas as pd
from pathlib import Path

import config
from src.physiological.model import PhysiologicalStressModel
from src.location.model import SpatialRiskEngine
from src.environment.model import EnvironmentalContextEngine

class TestBaseModels(unittest.TestCase):
    def setUp(self):
        self.physio_model = PhysiologicalStressModel()
        self.spatial_engine = SpatialRiskEngine()
        self.env_engine = EnvironmentalContextEngine()

    def test_physiological_model_output_range_and_semantics(self):
        """Tests that physiological stress model returns a valid probability in [0.0, 1.0]."""
        dummy_feats = {f: 0.5 for f in self.physio_model.feature_names}
        prob = self.physio_model.predict_proba(dummy_feats)
        self.assertIsInstance(prob, float)
        self.assertGreaterEqual(prob, 0.0)
        self.assertLessEqual(prob, 1.0)
        
        meta = self.physio_model.get_metadata()
        self.assertEqual(meta["branch"], "Physiological")
        self.assertIn("WESAD", meta["target_semantics"])

    def test_physiological_missing_feature_imputation(self):
        """Tests that missing or NaN features are gracefully imputed without raising errors."""
        empty_feats = {}
        prob_empty = self.physio_model.predict_proba(empty_feats)
        self.assertIsInstance(prob_empty, float)
        self.assertGreaterEqual(prob_empty, 0.0)
        self.assertLessEqual(prob_empty, 1.0)

        nan_feats = {self.physio_model.feature_names[0]: float("nan")} if self.physio_model.feature_names else {}
        prob_nan = self.physio_model.predict_proba(nan_feats)
        self.assertIsInstance(prob_nan, float)

    def test_spatial_engine_deterministic_output(self):
        """Tests that spatial engine computes deterministic Haversine risk in [0.0, 1.0]."""
        # Test Main Gate (High security, distance=0m)
        main_gate = config.BIT_MESRA_LANDMARKS_DB["main_gate"]
        score_gate = self.spatial_engine.predict_score(
            lat=main_gate["latitude"],
            lon=main_gate["longitude"],
            time_of_day_hour=14.0
        )
        self.assertIsInstance(score_gate, float)
        self.assertGreaterEqual(score_gate, 0.0)
        self.assertLessEqual(score_gate, 1.0)
        self.assertLess(score_gate, 0.35, "Main gate should have low vulnerability during daytime.")

        # Test Hostel 13 (Isolated perimeter sector)
        h13 = config.BIT_MESRA_LANDMARKS_DB["h13"]
        score_h13 = self.spatial_engine.predict_score(
            lat=h13["latitude"],
            lon=h13["longitude"],
            time_of_day_hour=23.5 # Late night
        )
        self.assertGreater(score_h13, score_gate, "Isolated perimeter hostel at night must have higher risk than main gate.")

    def test_environmental_engine_sensitivity(self):
        """Tests environmental engine response across day/night, light/dark, low/high crowd."""
        # Daytime, Light, High crowd (lowest risk)
        score_safe = self.env_engine.predict_score(time_of_day_hour=14.0, lighting_condition="Light", crowd_density="High")
        # Late night, Dark, Low crowd (highest risk)
        score_danger = self.env_engine.predict_score(time_of_day_hour=2.0, lighting_condition="Dark", crowd_density="Low")

        self.assertGreaterEqual(score_safe, 0.05)
        self.assertLessEqual(score_danger, 1.0)
        self.assertGreater(score_danger, score_safe, "Night dark isolated conditions must evaluate higher risk than day lit public space.")

    def test_environmental_invalid_inputs(self):
        """Tests environmental engine robust handling of hour wrap-around and unconventional casing."""
        score_wrap = self.env_engine.predict_score(time_of_day_hour=26.5, lighting_condition="  DARK  ", crowd_density="low")
        self.assertIsInstance(score_wrap, float)
        self.assertGreaterEqual(score_wrap, 0.0)
        self.assertLessEqual(score_wrap, 1.0)

if __name__ == "__main__":
    unittest.main()

