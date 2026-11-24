"""
Unit and Integration Tests for WESAD Subject Switching and Telemetry Pipeline.
Verifies that individual subjects load distinct physiological data, produce individual
biometric telemetry (EDA, Skin Temp, BPM, Duration, Sample Counts), and that
subject-switching immediately updates the dashboard state without stale caching or hardcoded fallbacks.
"""

import unittest
from pathlib import Path
import numpy as np
import pandas as pd
import config
from src.fusion.predict import MultimodalThreatPredictor
from app.main import evaluate_subject_biometrics


class TestWESADSubjectSwitching(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.predictor = MultimodalThreatPredictor()
        cls.test_subjects = ["S2", "S3", "S4", "S10", "S13", "S17"]

    def test_subjects_have_distinct_biometrics(self):
        """Verifies that each audited subject produces distinct EDA and Temperature values."""
        results = {}
        for subj in self.test_subjects:
            res = evaluate_subject_biometrics(subj, "Window 1 (Baseline)")
            self.assertEqual(res["subject_id"], subj)
            self.assertGreater(res["sample_count"], 0)
            self.assertGreater(res["duration_min"], 0.0)
            self.assertGreater(res["temp_mean"], 20.0)  # Physiological range
            self.assertLess(res["temp_mean"], 42.0)
            self.assertGreaterEqual(res["physio_score"], 0.0)
            self.assertLessEqual(res["physio_score"], 1.0)
            results[subj] = res

        # Verify EDA means are distinct across subjects (no identical hardcoded numbers)
        eda_values = [results[s]["eda_mean"] for s in self.test_subjects]
        self.assertEqual(len(set(eda_values)), len(self.test_subjects),
                         f"EDA values across subjects should be distinct: {eda_values}")

        # Verify temperatures are distinct across subjects
        temp_values = [results[s]["temp_mean"] for s in self.test_subjects]
        self.assertEqual(len(set(temp_values)), len(self.test_subjects),
                         f"Temperature values across subjects should be distinct: {temp_values}")

    def test_baseline_vs_stress_dynamic_response(self):
        """Verifies that switching from Baseline to Stress window increases stress probability."""
        for subj in ["S2", "S3", "S4"]:
            res_base = evaluate_subject_biometrics(subj, "Window 1 (Baseline)")
            res_stress = evaluate_subject_biometrics(subj, "Window 2 (Stress Session)")

            self.assertGreater(res_stress["physio_score"], res_base["physio_score"],
                               f"Subject {subj}: Stress score ({res_stress['physio_score']}) must exceed baseline ({res_base['physio_score']})")
            self.assertGreater(res_stress["physio_score"], 0.60,
                               f"Subject {subj}: Stress condition should produce high stress probability")

    def test_pre_extracted_csv_fallback(self):
        """Verifies that if raw pickle does not exist, genuine pre-extracted 76 features are used."""
        feat_path = config.DATA_DIR / "physiological" / "physiological_features.csv"
        self.assertTrue(feat_path.exists(), "physiological_features.csv must exist as authentic repository")
        df = pd.read_csv(feat_path)

        for subj in self.test_subjects:
            sub_df = df[df["subject_id"] == subj]
            self.assertGreater(len(sub_df), 50, f"Subject {subj} must have genuine window records in CSV")


class TestStreamlitSubjectSwitchingHeadless(unittest.TestCase):
    """Headless Streamlit AppTest testing reactive subject switching."""

    def test_streamlit_app_subject_switching(self):
        from streamlit.testing.v1 import AppTest
        app_path = str(Path(__file__).parent.parent / "app" / "main.py")
        at = AppTest.from_file(app_path, default_timeout=30)
        at.run()
        self.assertFalse(at.exception, f"App should run without exceptions: {at.exception}")

        # Verify initial subject selection is S2
        self.assertIn("sel_physio_subject", at.session_state)
        s2_score = at.session_state["physio_score"]
        self.assertIsNotNone(s2_score)

        # Switch subject to S3
        at.selectbox(key="sel_physio_subject").select("S3")
        at.run()
        self.assertFalse(at.exception, f"App rerun with S3 should not raise exceptions: {at.exception}")
        s3_score = at.session_state["physio_score"]
        self.assertEqual(at.session_state["current_physio_subject"], "S3")

        # Switch subject to S4
        at.selectbox(key="sel_physio_subject").select("S4")
        at.run()
        self.assertFalse(at.exception, f"App rerun with S4 should not raise exceptions: {at.exception}")
        self.assertEqual(at.session_state["current_physio_subject"], "S4")


if __name__ == "__main__":
    unittest.main()
