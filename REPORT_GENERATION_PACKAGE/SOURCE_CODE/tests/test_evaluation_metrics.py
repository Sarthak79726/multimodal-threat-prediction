"""
Unit Tests for Evaluation Metrics, Brier Score, and Calibration Calculations.
Tests:
- evaluate_predictions returns complete metric suite (Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, Brier, ECE).
- Expected Calibration Error (ECE) calculation logic.
- Model comparison table formatting.
"""

import unittest
import numpy as np
from src.fusion.evaluation import (
    evaluate_predictions,
    compute_expected_calibration_error,
    format_comparison_table
)

class TestEvaluationMetrics(unittest.TestCase):
    def test_metrics_calculation(self):
        """Verifies metric calculations on controlled synthetic ground truth and predictions."""
        y_true = np.array([0, 0, 1, 1, 0, 1, 0, 0, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.8, 0.9, 0.4, 0.6, 0.3, 0.2, 0.7, 0.85])

        metrics = evaluate_predictions(y_true, y_prob, threshold=0.50, model_name="TestModel")

        self.assertEqual(metrics["model_name"], "TestModel")
        self.assertAlmostEqual(metrics["accuracy"], 1.0, places=2)
        self.assertAlmostEqual(metrics["precision"], 1.0, places=2)
        self.assertAlmostEqual(metrics["recall"], 1.0, places=2)
        self.assertAlmostEqual(metrics["f1_score"], 1.0, places=2)
        self.assertAlmostEqual(metrics["roc_auc"], 1.0, places=2)
        self.assertLess(metrics["brier_score"], 0.10)
        self.assertIsInstance(metrics["expected_calibration_error"], float)

    def test_expected_calibration_error(self):
        """Verifies that perfectly calibrated probabilities yield low ECE."""
        y_true = np.array([0, 0, 1, 1])
        # Perfectly matching empirical frequency
        y_prob = np.array([0.0, 0.0, 1.0, 1.0])
        ece, bins = compute_expected_calibration_error(y_true, y_prob, n_bins=5)
        self.assertEqual(ece, 0.0)

    def test_comparison_table_formatting(self):
        """Verifies comparison table generates a structured pandas DataFrame."""
        results = {
            "Model A": {"accuracy": 0.90, "precision": 0.85, "recall": 0.88, "f1_score": 0.86, "roc_auc": 0.95, "pr_auc": 0.90, "brier_score": 0.05, "expected_calibration_error": 0.03},
            "Model B": {"accuracy": 0.92, "precision": 0.89, "recall": 0.91, "f1_score": 0.90, "roc_auc": 0.97, "pr_auc": 0.92, "brier_score": 0.04, "expected_calibration_error": 0.02}
        }
        df = format_comparison_table(results)
        self.assertEqual(len(df), 2)
        self.assertIn("Model / Branch", df.columns)
        self.assertIn("ROC-AUC", df.columns)
        self.assertIn("Brier Score", df.columns)

if __name__ == "__main__":
    unittest.main()

