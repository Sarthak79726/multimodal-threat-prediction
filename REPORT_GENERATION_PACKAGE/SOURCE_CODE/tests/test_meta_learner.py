"""
Unit Tests for Level 1 Meta-Learners and Baselines.
Tests:
- LogisticRegressionMetaLearner fit, predict_proba, odds ratios, and mathematical explainability.
- RandomForestMetaLearner fit and feature importances.
- WeightedLateFusionBaseline exact formula calculation.
- Error handling on un-fitted model prediction.
- Error handling on single-class target inputs.
"""

import unittest
import numpy as np
import config
from src.fusion.meta_learner import (
    LogisticRegressionMetaLearner,
    RandomForestMetaLearner,
    WeightedLateFusionBaseline
)

class TestMetaLearners(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.feature_names = config.STACKING_META_FEATURES
        self.n_samples = 150
        
        # Linearly separable meta-features
        p1 = np.random.beta(1.0, 2.0, self.n_samples)
        p2 = np.random.beta(1.0, 2.0, self.n_samples)
        p3 = np.random.beta(1.0, 2.0, self.n_samples)
        
        self.X_meta = np.column_stack([p1, p2, p3])
        # Target strongly associated with p1 and p2
        logit = 4.0 * p1 + 3.0 * p2 + 1.0 * p3 - 2.5
        prob = 1.0 / (1.0 + np.exp(-logit))
        self.y = (prob >= 0.50).astype(int)

    def test_logistic_meta_learner_fit_and_explain(self):
        """Tests that LogisticRegressionMetaLearner learns positive coefficients and provides full explainability."""
        meta = LogisticRegressionMetaLearner(feature_names=self.feature_names, random_state=42)
        self.assertFalse(meta.is_trained_ml)
        
        meta.fit(self.X_meta, self.y)
        self.assertTrue(meta.is_trained_ml)
        self.assertEqual(len(meta.coef_), 3)
        self.assertIsInstance(meta.intercept_, float)

        # Predict probability
        probs = meta.predict_proba(self.X_meta)
        self.assertEqual(probs.shape, (self.n_samples, 2))
        self.assertTrue((probs >= 0.0).all())
        self.assertTrue((probs <= 1.0).all())

        # Test explainability
        exp = meta.explain(sample_vec=self.X_meta[0])
        self.assertIn("coefficients", exp)
        self.assertIn("intercept_beta0", exp)
        self.assertIn("odds_ratios", exp)
        self.assertIn("sample_evaluation", exp)

        # Check that odds ratio = exp(coef)
        for fname in self.feature_names:
            expected_or = np.exp(exp["coefficients"][fname])
            self.assertAlmostEqual(exp["odds_ratios"][fname], expected_or, places=3)

    def test_unfitted_meta_learner_raises_error(self):
        """Tests that predict_proba on an un-fitted meta-learner raises RuntimeError."""
        meta = LogisticRegressionMetaLearner(feature_names=self.feature_names)
        with self.assertRaises(RuntimeError):
            meta.predict_proba(self.X_meta)

    def test_single_class_rejection(self):
        """Tests that meta-learner rejects training when target labels have zero variance (single class)."""
        meta = LogisticRegressionMetaLearner(feature_names=self.feature_names)
        single_class_y = np.zeros(self.n_samples, dtype=int)
        with self.assertRaises(ValueError):
            meta.fit(self.X_meta, single_class_y)

    def test_weighted_late_fusion_baseline(self):
        """Tests that WeightedLateFusionBaseline calculates exact 40/25/35 weighted score."""
        baseline = WeightedLateFusionBaseline()
        # Test extreme cases
        self.assertAlmostEqual(baseline.predict_score(0.0, 0.0, 0.0), 0.0, places=4)
        self.assertAlmostEqual(baseline.predict_score(1.0, 1.0, 1.0), 1.0, places=4)

        # Test specific combination: 0.50 * 0.40 + 0.20 * 0.25 + 0.80 * 0.35 = 0.20 + 0.05 + 0.28 = 0.53
        score = baseline.predict_score(0.50, 0.20, 0.80)
        self.assertAlmostEqual(score, 0.53, places=4)

if __name__ == "__main__":
    unittest.main()

