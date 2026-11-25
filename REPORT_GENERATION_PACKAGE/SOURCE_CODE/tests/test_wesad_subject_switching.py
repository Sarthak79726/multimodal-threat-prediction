"""
Unit and Integration Tests for WESAD Subject & Condition Isolation.
Verifies that individual subjects and conditions (Baseline, Stress, Amusement, Meditation)
load their own authentic recordings, produce unique telemetry (EDA, Skin Temp, BVP, Fingerprints),
and that changing subject or condition dynamically updates the application without stale caching
or global aggregation.
"""

import unittest
from pathlib import Path
import json
import numpy as np
import pandas as pd
import config
from src.fusion.predict import MultimodalThreatPredictor
from app.main import evaluate_subject_biometrics


class TestWESADSubjectSwitching(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.predictor = MultimodalThreatPredictor()
        cls.test_subjects = ["S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10", "S11", "S13", "S14", "S15", "S16", "S17"]

    def test_s2_and_s3_baseline_are_distinct(self):
        """1 & 2: Verifies S2 Baseline and S3 Baseline load from respective subjects with distinct data."""
        s2_base = evaluate_subject_biometrics("S2", "Window 1 (Baseline)")
        s3_base = evaluate_subject_biometrics("S3", "Window 1 (Baseline)")

        self.assertEqual(s2_base["subject_id"], "S2")
        self.assertEqual(s3_base["subject_id"], "S3")
        self.assertNotEqual(s2_base["fingerprint"], s3_base["fingerprint"],
                            "S2 Baseline and S3 Baseline must have distinct fingerprints")
        self.assertNotEqual(s2_base["eda_mean"], s3_base["eda_mean"],
                            "S2 and S3 Baseline must have distinct EDA means")
        self.assertNotEqual(s2_base["temp_mean"], s3_base["temp_mean"],
                            "S2 and S3 Baseline must have distinct Temperature means")

    def test_s2_and_s3_amusement_are_distinct(self):
        """3 & 4: Verifies S2 Amusement and S3 Amusement load from respective subjects with distinct data."""
        s2_amu = evaluate_subject_biometrics("S2", "Window 3 (Amusement)")
        s3_amu = evaluate_subject_biometrics("S3", "Window 3 (Amusement)")

        self.assertEqual(s2_amu["subject_id"], "S2")
        self.assertEqual(s3_amu["subject_id"], "S3")
        self.assertNotEqual(s2_amu["fingerprint"], s3_amu["fingerprint"],
                            "S2 Amusement and S3 Amusement must have distinct fingerprints")
        self.assertNotEqual(s2_amu["eda_mean"], s3_amu["eda_mean"],
                            "S2 and S3 Amusement must have distinct EDA means")
        self.assertNotEqual(s2_amu["temp_mean"], s3_amu["temp_mean"],
                            "S2 and S3 Amusement must have distinct Temperature means")

    def test_s3_and_s4_baseline_and_amusement_distinct(self):
        """Direct comparison between S3 and S4 across Baseline and Amusement."""
        s3_base = evaluate_subject_biometrics("S3", "Window 1 (Baseline)")
        s4_base = evaluate_subject_biometrics("S4", "Window 1 (Baseline)")
        self.assertNotEqual(s3_base["fingerprint"], s4_base["fingerprint"])
        self.assertNotEqual(s3_base["eda_mean"], s4_base["eda_mean"])

        s3_amu = evaluate_subject_biometrics("S3", "Window 3 (Amusement)")
        s4_amu = evaluate_subject_biometrics("S4", "Window 3 (Amusement)")
        self.assertNotEqual(s3_amu["fingerprint"], s4_amu["fingerprint"])
        self.assertNotEqual(s3_amu["eda_mean"], s4_amu["eda_mean"])

    def test_condition_switching_within_subject_updates_data(self):
        """5 & 6: Verifies switching between Baseline and Amusement within a subject updates telemetry."""
        for subj in ["S2", "S3", "S4"]:
            res_base = evaluate_subject_biometrics(subj, "Window 1 (Baseline)")
            res_amu = evaluate_subject_biometrics(subj, "Window 3 (Amusement)")
            res_stress = evaluate_subject_biometrics(subj, "Window 2 (Stress Session)")

            # Each condition within subject must have unique fingerprint
            fps = {res_base["fingerprint"], res_amu["fingerprint"], res_stress["fingerprint"]}
            self.assertEqual(len(fps), 3, f"Subject {subj}: All 3 conditions must have unique fingerprints: {fps}")

            # Stress score must exceed baseline score
            self.assertGreater(res_stress["physio_score"], res_base["physio_score"])

    def test_no_global_condition_vector_reuse(self):
        """7: Verifies all audited subjects and conditions possess unique SHA-256 fingerprints."""
        audit_path = config.DATA_DIR / "physiological" / "wesad_condition_audit.json"
        self.assertTrue(audit_path.exists(), "wesad_condition_audit.json must exist")
        with open(audit_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        fingerprints = []
        for entry in data:
            s_id = entry["subject_id"]
            for cond_name, c_data in entry["conditions"].items():
                if c_data.get("available"):
                    fp = c_data["fingerprint"]
                    fingerprints.append((s_id, cond_name, fp))

        unique_fps = set(x[2] for x in fingerprints)
        self.assertEqual(len(fingerprints), len(unique_fps),
                         f"Detected duplicate fingerprint among {len(fingerprints)} condition segments!")

    def test_signal_label_alignment(self):
        """8: Verifies that physiological signal lengths and label array lengths are strictly identical."""
        audit_path = config.DATA_DIR / "physiological" / "wesad_condition_audit.json"
        with open(audit_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for entry in data:
            self.assertTrue(entry["alignment_valid"],
                            f"Subject {entry['subject_id']} has mismatched label/signal alignment!")

    def test_meditation_never_fifty_percent_across_subjects(self):
        """9: Verifies that Meditation stress probability evaluates to authentic Non-Stress (<10%) and NEVER 50%."""
        audit_path = config.DATA_DIR / "physiological" / "wesad_condition_audit.json"
        with open(audit_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for entry in data:
            s_id = entry["subject_id"]
            med_info = entry["conditions"]["Meditation"]
            self.assertTrue(med_info.get("available"), f"Subject {s_id} Meditation must be available")
            prob = med_info["stress_probability"]
            # Must not be the fallback 0.50 (50%)
            self.assertNotEqual(prob, 0.50, f"Subject {s_id} Meditation evaluated to fallback 50%!")
            # Non-Stress clinical state must have low stress probability (< 10%)
            self.assertLess(prob, 0.10, f"Subject {s_id} Meditation stress prob must be < 10%, got {prob*100:.2f}%")
            self.assertEqual(med_info.get("predicted_class"), "NON-STRESS")

    def test_baseline_and_amusement_distinct_telemetry_and_unrounded_display(self):
        """10: Verifies Baseline and Amusement have distinct unrounded probabilities and telemetry across subjects."""
        for subj in ["S2", "S3", "S4", "S5"]:
            base = evaluate_subject_biometrics(subj, "Window 1 (Baseline)")
            amu = evaluate_subject_biometrics(subj, "Window 3 (Amusement)")
            self.assertNotEqual(base["fingerprint"], amu["fingerprint"])
            self.assertNotEqual(base["eda_mean"], amu["eda_mean"])
            self.assertNotEqual(base["physio_score"], amu["physio_score"],
                                f"Subject {subj}: Baseline and Amusement unrounded scores must be distinct")
            # Verify predicted_class is present and classified as NON-STRESS
            self.assertEqual(base["predicted_class"], "NON-STRESS")
            self.assertEqual(amu["predicted_class"], "NON-STRESS")


class TestStreamlitSubjectSwitchingHeadless(unittest.TestCase):
    """Headless Streamlit AppTest testing reactive subject and condition switching."""

    def test_streamlit_app_subject_and_condition_switching(self):
        from streamlit.testing.v1 import AppTest
        app_path = str(Path(__file__).parent.parent / "app" / "main.py")
        at = AppTest.from_file(app_path, default_timeout=30)
        at.run()
        self.assertFalse(at.exception, f"App should run without exceptions: {at.exception}")

        # 1. Initial subject is S2, condition is Window 1 (Baseline)
        self.assertIn("sel_physio_subject", at.session_state)
        s2_base_telemetry = at.session_state["physio_telemetry"]
        self.assertIsNotNone(s2_base_telemetry)
        self.assertEqual(s2_base_telemetry["subject_id"], "S2")
        self.assertEqual(s2_base_telemetry["condition_name"], "Baseline")
        s2_base_fp = s2_base_telemetry["fingerprint"]

        # 2. Switch condition to Amusement
        at.selectbox(key="sel_physio_window").select("Window 3 (Amusement)")
        at.run()
        self.assertFalse(at.exception)
        s2_amu_telemetry = at.session_state["physio_telemetry"]
        self.assertEqual(s2_amu_telemetry["condition_name"], "Amusement")
        self.assertNotEqual(s2_amu_telemetry["fingerprint"], s2_base_fp,
                            "S2 Amusement must have a distinct fingerprint from S2 Baseline")

        # 3. Switch subject to S3
        at.selectbox(key="sel_physio_subject").select("S3")
        at.run()
        self.assertFalse(at.exception)
        s3_amu_telemetry = at.session_state["physio_telemetry"]
        self.assertEqual(s3_amu_telemetry["subject_id"], "S3")
        self.assertEqual(s3_amu_telemetry["condition_name"], "Amusement")
        self.assertNotEqual(s3_amu_telemetry["fingerprint"], s2_amu_telemetry["fingerprint"],
                            "S3 Amusement must have a distinct fingerprint from S2 Amusement")

        # 4. Switch condition to Window 4 (Meditation)
        at.selectbox(key="sel_physio_window").select("Window 4 (Meditation)")
        at.run()
        self.assertFalse(at.exception)
        s3_med_telemetry = at.session_state["physio_telemetry"]
        self.assertEqual(s3_med_telemetry["condition_name"], "Meditation")
        med_score = at.session_state["physio_score"]
        self.assertNotEqual(med_score, 0.50, "Meditation stress probability must NOT be 50%!")
        self.assertLess(med_score, 0.10, f"Meditation probability must be < 10%, got {med_score*100:.2f}%")
        self.assertEqual(s3_med_telemetry["predicted_class"], "NON-STRESS")


if __name__ == "__main__":
    unittest.main()
