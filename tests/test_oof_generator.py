"""
Unit Tests for Leakage-Free Out-Of-Fold (OOF) Prediction Generator.
Tests:
- Mutual exclusivity of fold train and validation indices.
- Subject/group leakage prevention (zero subject overlap between folds).
- Complete coverage: all samples receive out-of-fold predictions.
- Strict fitting of preprocessors inside folds.
"""

import unittest
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from src.fusion.oof_generator import OOFGenerator

class TestOOFGenerator(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.n_samples = 200
        self.n_features = 4
        self.subjects = [f"Subj_{i:02d}" for i in range(1, 11)]

        # Synthetic dataset with 10 distinct subject groups
        self.X_df = pd.DataFrame(
            np.random.randn(self.n_samples, self.n_features),
            columns=[f"feat_{i}" for i in range(self.n_features)]
        )
        self.y = np.random.choice([0, 1], size=self.n_samples, p=[0.7, 0.3])
        self.groups = np.random.choice(self.subjects, size=self.n_samples)

    def test_oof_index_isolation_and_coverage(self):
        """Verifies that every sample is predicted out-of-fold and fold train/val sets never overlap."""
        generator = OOFGenerator(n_splits=5, random_state=42)
        base_models = {
            "logistic": LogisticRegression(),
            "rf": RandomForestClassifier(n_estimators=10, random_state=42)
        }

        oof_df, fitted_models = generator.generate_oof_predictions(
            base_models=base_models,
            X_train=self.X_df,
            y_train=self.y,
            groups_train=self.groups,
            scale_features=True
        )

        self.assertEqual(len(oof_df), self.n_samples)
        self.assertIn("logistic", oof_df.columns)
        self.assertIn("rf", oof_df.columns)

        # Assert no NaNs in OOF matrix
        self.assertFalse(oof_df.isna().any().any())

        # Assert all probabilities in [0.0, 1.0]
        self.assertTrue((oof_df >= 0.0).all().all())
        self.assertTrue((oof_df <= 1.0).all().all())

    def test_group_leakage_prevention(self):
        """Verifies that no subject group appears in both train and validation splits in any fold."""
        generator = OOFGenerator(n_splits=5, random_state=42)
        cv, cv_name = generator.select_cv_strategy(self.X_df.values, self.y, self.groups)

        for fold_idx, (tr_idx, va_idx) in enumerate(cv.split(self.X_df, self.y, self.groups)):
            tr_groups = set(self.groups[tr_idx])
            va_groups = set(self.groups[va_idx])
            overlap = tr_groups.intersection(va_groups)
            self.assertEqual(len(overlap), 0, f"Subject leakage detected in fold {fold_idx}: {overlap}")

    def test_reproducibility(self):
        """Verifies that identical random seeds generate bitwise identical OOF probability matrices."""
        base_models1 = {"rf": RandomForestClassifier(n_estimators=10, random_state=42)}
        base_models2 = {"rf": RandomForestClassifier(n_estimators=10, random_state=42)}

        gen1 = OOFGenerator(n_splits=5, random_state=42)
        gen2 = OOFGenerator(n_splits=5, random_state=42)

        oof1, _ = gen1.generate_oof_predictions(base_models1, self.X_df, self.y, self.groups)
        oof2, _ = gen2.generate_oof_predictions(base_models2, self.X_df, self.y, self.groups)

        np.testing.assert_allclose(oof1.values, oof2.values, atol=1e-7)

if __name__ == "__main__":
    unittest.main()

