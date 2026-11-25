"""
Probability Calibration, Reliability Curves, and Uncertainty Metrics.
Implements calibration evaluation (Platt Scaling, Isotonic Regression),
Brier score computation, Expected Calibration Error (ECE), and reliability diagram generation.
Enforces the principle that heuristic or weighted indices are not genuine probabilities
without empirical ground-truth validation.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss

class ProbabilityCalibrator:
    """
    Calibrates continuous model scores into well-calibrated posterior probabilities.
    Supports:
    - Platt Scaling (Sigmoid / Logistic regression)
    - Non-parametric Isotonic Regression
    """

    def __init__(self, method: str = "platt", random_state: int = 42):
        self.method = method.lower()
        self.random_state = random_state
        self.is_calibrated: bool = False
        self.calibrator = None
        self.brier_score_val: Optional[float] = None
        self.ece_val: Optional[float] = None

    def fit(self, scores: np.ndarray, y_true: np.ndarray) -> "ProbabilityCalibrator":
        """Fits calibration curve on validation scores and ground-truth binary outcomes."""
        s_arr = np.asarray(scores, dtype=np.float64).ravel()
        y_arr = np.asarray(y_true, dtype=np.int32).ravel()

        if len(s_arr) != len(y_arr):
            raise ValueError(f"Length mismatch: {len(s_arr)} scores vs {len(y_arr)} labels.")
        if len(np.unique(y_arr)) < 2:
            raise ValueError("Calibration requires at least two target classes.")

        if self.method == "platt":
            # Platt Scaling: Logistic Regression on log-odds or raw score
            clf = LogisticRegression(C=1.0, solver="lbfgs", random_state=self.random_state)
            clf.fit(s_arr.reshape(-1, 1), y_arr)
            self.calibrator = clf
        elif self.method == "isotonic":
            # Isotonic Regression: non-parametric isotonic fit
            iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            iso.fit(s_arr, y_arr)
            self.calibrator = iso
        else:
            raise ValueError(f"Unknown calibration method: {self.method}")

        self.is_calibrated = True

        # Compute post-calibration Brier score and ECE on training/val set
        cal_probs = self.predict_proba(s_arr)
        self.brier_score_val = float(brier_score_loss(y_arr, cal_probs))
        diag = self.compute_calibration_curve(y_arr, cal_probs)
        self.ece_val = diag["expected_calibration_error"]

        return self

    def predict_proba(self, scores: np.ndarray) -> np.ndarray:
        """Transforms uncalibrated raw scores into calibrated probabilities."""
        s_arr = np.asarray(scores, dtype=np.float64).ravel()
        if not self.is_calibrated:
            # If not calibrated, return input clipped to [0, 1] without pretending it is calibrated
            return np.clip(s_arr, 0.0, 1.0)

        if self.method == "platt":
            return self.calibrator.predict_proba(s_arr.reshape(-1, 1))[:, 1]
        else:
            return self.calibrator.predict(s_arr)

    @staticmethod
    def compute_calibration_curve(
        y_true: np.ndarray,
        y_prob: np.ndarray,
        n_bins: int = 10
    ) -> Dict[str, Any]:
        """
        Computes reliability diagram statistics:
        - Bin midpoints
        - Mean predicted probability per bin
        - Fraction of positives (true empirical accuracy) per bin
        - Expected Calibration Error (ECE)
        - Maximum Calibration Error (MCE)
        """
        y_t = np.asarray(y_true, dtype=np.int32).ravel()
        y_p = np.asarray(y_prob, dtype=np.float64).ravel()

        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
        bin_accuracies = []
        bin_confidences = []
        bin_counts = []
        ece = 0.0
        mce = 0.0
        n_total = len(y_t)

        for i in range(n_bins):
            b_start, b_end = bin_edges[i], bin_edges[i + 1]
            if i == n_bins - 1:
                mask = (y_p >= b_start) & (y_p <= b_end)
            else:
                mask = (y_p >= b_start) & (y_p < b_end)

            count = int(np.sum(mask))
            bin_counts.append(count)

            if count > 0:
                acc = float(np.mean(y_t[mask]))
                conf = float(np.mean(y_p[mask]))
                diff = abs(acc - conf)
                ece += (count / n_total) * diff
                mce = max(mce, diff)
                bin_accuracies.append(round(acc, 4))
                bin_confidences.append(round(conf, 4))
            else:
                bin_accuracies.append(None)
                bin_confidences.append(None)

        return {
            "n_bins": n_bins,
            "bin_edges": [round(float(e), 2) for e in bin_edges],
            "bin_counts": bin_counts,
            "bin_accuracies": bin_accuracies,
            "bin_confidences": bin_confidences,
            "expected_calibration_error": round(float(ece), 4),
            "maximum_calibration_error": round(float(mce), 4),
            "brier_score": round(float(brier_score_loss(y_t, y_p)), 4) if len(y_t) > 0 else None
        }
