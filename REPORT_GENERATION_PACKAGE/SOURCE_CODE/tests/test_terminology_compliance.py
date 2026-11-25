"""
Terminology Compliance and Scientific Integrity Regression Tests.
Prevents scientifically misleading terminology from appearing in the application:
- Rejects labels: 'SAFE', 'UNSAFE', 'THREAT DETECTED', 'THREAT RISK SCORE',
  'PROTOTYPE THREAT-RISK SCORE', 'NORMAL SAFE PARAMETERS'.
- Enforces neutral context classification: 'STANDARD CONTEXT', 'ELEVATED CONTEXT', 'HIGH CONTEXT'.
- Validates the exact 32.00% benchmark scenario and decision rationale.
"""

import unittest
import re
from pathlib import Path

import config
from src.fusion.predict import MultimodalThreatPredictor

class TestTerminologyCompliance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.predictor = MultimodalThreatPredictor()

    def test_user_benchmark_scenario_32_percent(self):
        """
        Validates the exact user prompt scenario:
        - Phys: 18% x 0.40 = 7.20%
        - Loc:  60% x 0.25 = 15.00%
        - Env:  28% x 0.35 = 9.80%
        - Final: 32.00%
        - Context Classification: STANDARD CONTEXT
        - Meta-Learner: Not trained
        """
        res = self.predictor.predict_fusion(
            physiological_stress_score=0.18,
            location_risk_score=0.60,
            environment_risk_score=0.28
        )

        # 1. Calculation verification
        self.assertAlmostEqual(res["experimental_context_index"], 0.32, places=4)
        self.assertAlmostEqual(res["experimental_context_index_pct"], 32.00, places=2)
        self.assertAlmostEqual(res["contributions_pct"]["physiological"], 7.20, places=2)
        self.assertAlmostEqual(res["contributions_pct"]["location"], 15.00, places=2)
        self.assertAlmostEqual(res["contributions_pct"]["environment"], 9.80, places=2)

        # 2. Context Classification
        self.assertEqual(res["context_status"], "STANDARD CONTEXT")
        self.assertEqual(res["context_classification"], "STANDARD CONTEXT")

        # 3. Meta-Learner Status
        self.assertFalse(res["is_meta_learner_trained"])
        self.assertEqual(res["meta_model_status"], "UNTRAINED — VALID MULTIMODAL THREAT DATASET REQUIRED")

        # 4. Exact Decision Rationale
        expected_rationale = (
            "Context Assessment: The experimental weighted context index is 32.00%. "
            "It combines physiological stress (18.0%), location context risk (60.0%) "
            "and environmental context risk (28.0%) using predefined reference weights. "
            "This value describes contextual inputs only and does not establish whether a person or location is safe or unsafe."
        )
        self.assertEqual(res["explanation"], expected_rationale)

        # 5. Explanatory Note
        self.assertEqual(
            res["clarification"],
            "This index represents a heuristic combination of physiological, spatial and environmental context scores. It is not a validated threat prediction."
        )

    def test_neutral_context_classification_thresholds(self):
        """Verifies neutral context classification across configured project thresholds."""
        # Standard context (<= 0.39)
        res_low = self.predictor.predict_fusion(0.10, 0.20, 0.15)
        self.assertEqual(res_low["context_status"], "STANDARD CONTEXT")

        # Elevated context (0.39 < score <= 0.69)
        res_mid = self.predictor.predict_fusion(0.50, 0.60, 0.45)
        self.assertEqual(res_mid["context_status"], "ELEVATED CONTEXT")

        # High context (> 0.69)
        res_high = self.predictor.predict_fusion(0.85, 0.90, 0.80)
        self.assertEqual(res_high["context_status"], "HIGH CONTEXT")

    def test_forbidden_terminology_absent_from_inference_outputs(self):
        """Verifies that forbidden misleading labels never appear in predictor outputs."""
        forbidden_regexes = [
            re.compile(r"\bSAFE\b"),
            re.compile(r"\bUNSAFE\b"),
            re.compile(r"THREAT DETECTED", re.IGNORECASE),
            re.compile(r"THREAT RISK SCORE", re.IGNORECASE),
            re.compile(r"PROTOTYPE THREAT-RISK SCORE", re.IGNORECASE),
            re.compile(r"NORMAL SAFE PARAMETERS", re.IGNORECASE)
        ]

        test_inputs = [
            (0.10, 0.10, 0.10),
            (0.18, 0.60, 0.28),
            (0.50, 0.50, 0.50),
            (0.90, 0.90, 0.90)
        ]

        for p, l, e in test_inputs:
            res = self.predictor.predict_fusion(p, l, e)
            status_str = f"{res['threat_status']} {res['context_status']} {res['context_classification']}"
            
            for reg in forbidden_regexes:
                self.assertIsNone(
                    reg.search(status_str),
                    f"Forbidden term '{reg.pattern}' found in status: '{status_str}' for inputs ({p}, {l}, {e})"
                )

    def test_dashboard_templates_do_not_contain_forbidden_labels(self):
        """Verifies that frontend templates and dashboard code do not contain forbidden misleading labels."""
        forbidden_phrases = [
            "PROTOTYPE THREAT-RISK SCORE",
            "THREAT DETECTED",
            "NORMAL SAFE PARAMETERS"
        ]

        dashboard_files = [
            Path("app/main.py"),
            Path("app/templates/dashboard.html")
        ]

        for fpath in dashboard_files:
            if fpath.exists():
                text = fpath.read_text(encoding="utf-8")
                for phrase in forbidden_phrases:
                    self.assertNotIn(
                        phrase, text.upper(),
                        f"Forbidden phrase '{phrase}' found in {fpath}"
                    )

if __name__ == "__main__":
    unittest.main()
