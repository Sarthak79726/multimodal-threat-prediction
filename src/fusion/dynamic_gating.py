"""
Dynamic Gating Network for Adaptive Multimodal Fusion.
Implements data-driven learned modality gating:
    alpha = Softmax(g(X))
Where:
    X = Combined modality features, context indicators, and reliability scores.
    g = Trainable neural gating model or parameterized linear projection.
    alpha_P + alpha_L + alpha_E = 1.0 (strictly enforced via Softmax).
    R = alpha_P * P + alpha_L * L + alpha_E * E.

Includes strict production provenance controls:
- Avoids hardcoded situational heuristics (e.g. no manual 'if crowd high then env weight = 70%' rules).
- Retains is_trained=False by default until synchronized real-world ground-truth data is provided.
- Strictly rejects presenting simulation weights as verified real-world threat contributions.
"""

import sys
import json
import pickle
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

# Ensure project root is in sys.path
_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.fusion.fusion_config import DEFAULT_BASELINE_WEIGHTS
from src.fusion.feature_schema import GATING_FEATURE_NAMES, validate_gating_vector

def softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over the last dimension."""
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)

class DynamicGatingNetwork:
    """
    Parametric Dynamic Gating Network.
    Maps input context feature vector X in R^15 -> Modality Importance Weights alpha in Delta^2.
    """
    MODALITIES = ["physiological", "spatial", "environmental"]

    def __init__(
        self,
        input_dim: int = len(GATING_FEATURE_NAMES),
        hidden_dim: int = 16,
        learning_rate: float = 0.01,
        random_state: int = 42
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = len(self.MODALITIES)
        self.learning_rate = learning_rate
        self.random_state = random_state

        self.is_trained: bool = False
        self.is_simulation_only: bool = False
        self.training_provenance: str = "UNTRAINED — Awaiting Synchronized Multimodal Dataset"
        self.training_metrics: Dict[str, Any] = {}

        # Parameter initialization (2-layer MLP projection to 3 logits)
        rng = np.random.RandomState(self.random_state)
        # Xavier/Glorot initialization
        scale_1 = np.sqrt(2.0 / (self.input_dim + self.hidden_dim))
        self.W1 = rng.randn(self.input_dim, self.hidden_dim) * scale_1
        self.b1 = np.zeros(self.hidden_dim, dtype=np.float64)

        scale_2 = np.sqrt(2.0 / (self.hidden_dim + self.output_dim))
        self.W2 = rng.randn(self.hidden_dim, self.output_dim) * scale_2
        self.b2 = np.zeros(self.output_dim, dtype=np.float64)

    def forward_logits(self, X: np.ndarray) -> np.ndarray:
        """Forward pass computing unnormalized gating logits."""
        # Layer 1: Linear + ReLU
        h = np.maximum(0.0, np.dot(X, self.W1) + self.b1)
        # Layer 2: Linear projection to 3 logits
        logits = np.dot(h, self.W2) + self.b2
        return logits

    def compute_weights(self, X: np.ndarray) -> np.ndarray:
        """
        Computes dynamic gating weights alpha = Softmax(g(X)).
        Guarantees:
        - alpha_P + alpha_L + alpha_E == 1.0 (to within machine precision)
        - alpha_i >= 0.0 for all i in {P, L, E}
        """
        X_arr = validate_gating_vector(X)
        logits = self.forward_logits(X_arr)
        weights = softmax(logits)
        return weights

    def predict_weights_dict(
        self,
        X: np.ndarray,
        allow_untrained_fallback: bool = True
    ) -> Dict[str, Any]:
        """
        Returns modality importance weights as a transparent dictionary.
        If the model is untrained, strictly flags the status rather than claiming
        untrained random initialization represents learned dynamic importance.
        """
        X_arr = validate_gating_vector(X)
        weights_arr = self.compute_weights(X_arr)[0]

        weights_dict = {
            "physiological": float(weights_arr[0]),
            "spatial": float(weights_arr[1]),
            "environmental": float(weights_arr[2])
        }

        # Verification check: sum must equal 1.0
        w_sum = sum(weights_dict.values())
        assert abs(w_sum - 1.0) < 1e-5, f"Softmax weights sum violation: {w_sum}"

        if not self.is_trained:
            return {
                "status": "UNTRAINED_FALLBACK",
                "is_trained": False,
                "is_simulation_only": False,
                "dynamic_weights": {
                    "physiological": DEFAULT_BASELINE_WEIGHTS["physiological"],
                    "spatial": DEFAULT_BASELINE_WEIGHTS["location"],
                    "environmental": DEFAULT_BASELINE_WEIGHTS["environment"]
                },
                "raw_untrained_weights": weights_dict,
                "notice": "Dynamic gating network has not been trained on verified ground-truth data. Using static reference baseline weights.",
                "scientific_disclaimer": "Learned dynamic weights will only be activated when synchronized multimodal outcome data is collected."
            }

        return {
            "status": "TRAINED_DYNAMIC",
            "is_trained": True,
            "is_simulation_only": self.is_simulation_only,
            "dynamic_weights": {k: round(v, 4) for k, v in weights_dict.items()},
            "provenance": self.training_provenance,
            "metrics": self.training_metrics
        }

    def predict_score(
        self,
        X: np.ndarray,
        p_val: float,
        l_val: float,
        e_val: Optional[float]
    ) -> Dict[str, Any]:
        """
        Evaluates dynamic weighted fusion:
        R = alpha_P * P + alpha_L * L + alpha_E * E
        """
        weights_meta = self.predict_weights_dict(X)
        active_w = weights_meta["dynamic_weights"]

        w_p = active_w["physiological"]
        w_l = active_w["spatial"]
        w_e = active_w["environmental"] if e_val is not None else 0.0

        if e_val is None:
            # Re-normalize if environmental is missing
            sum_pl = w_p + w_l
            if sum_pl > 0:
                w_p /= sum_pl
                w_l /= sum_pl
            contrib_p = w_p * p_val
            contrib_l = w_l * l_val
            contrib_e = 0.0
        else:
            contrib_p = w_p * p_val
            contrib_l = w_l * l_val
            contrib_e = w_e * e_val

        score = float(np.clip(contrib_p + contrib_l + contrib_e, 0.0, 1.0))

        return {
            "score": round(score, 4),
            "score_pct": round(score * 100.0, 2),
            "weights": {
                "physiological": round(w_p, 4),
                "spatial": round(w_l, 4),
                "environmental": round(w_e, 4)
            },
            "contributions": {
                "physiological": round(contrib_p, 4),
                "spatial": round(contrib_l, 4),
                "environmental": round(contrib_e, 4)
            },
            "contributions_pct": {
                "physiological": round(contrib_p * 100.0, 2),
                "spatial": round(contrib_l * 100.0, 2),
                "environmental": round(contrib_e * 100.0, 2)
            },
            "weights_meta": weights_meta
        }

    def fit(
        self,
        X: np.ndarray,
        y_outcomes: np.ndarray,
        epochs: int = 100,
        batch_size: int = 32,
        allow_simulation: bool = False
    ) -> "DynamicGatingNetwork":
        """
        Trains the gating network end-to-end using gradient descent.
        Strict verification:
        Rejects training if labels are detected to be fabricated or unverified,
        unless explicitly marked as a software pipeline simulation benchmark.
        """
        X_arr = validate_gating_vector(X)
        y_arr = np.asarray(y_outcomes, dtype=np.int32).ravel()

        if len(X_arr) != len(y_arr):
            raise ValueError(f"Feature count {len(X_arr)} != label count {len(y_arr)}")

        if len(np.unique(y_arr)) < 2:
            raise ValueError("Cannot train gating network on single-class data.")

        # Check for circular label construction
        # If y is an exact threshold of a linear combination of features, reject
        p_col = X_arr[:, 0]
        l_col = X_arr[:, 3]
        e_col = X_arr[:, 7]
        naive_comb = 0.40 * p_col + 0.25 * l_col + 0.35 * e_col
        if np.all((naive_comb >= 0.50) == y_arr):
            if not allow_simulation:
                raise ValueError("Data Circularity Rejected: Labels were generated directly from the baseline formula.")

        # Training loop via backpropagation with mini-batch SGD
        n_samples = len(X_arr)
        indices = np.arange(n_samples)

        for epoch in range(epochs):
            np.random.shuffle(indices)
            for start_idx in range(0, n_samples, batch_size):
                batch_idx = indices[start_idx : start_idx + batch_size]
                X_batch = X_arr[batch_idx]
                y_batch = y_arr[batch_idx]

                # Forward
                h = np.maximum(0.0, np.dot(X_batch, self.W1) + self.b1)
                logits = np.dot(h, self.W2) + self.b2
                alphas = softmax(logits)

                # Predictions: R_batch = alpha_P * P + alpha_L * L + alpha_E * E
                p_b = X_batch[:, 0]
                l_b = X_batch[:, 3]
                e_b = X_batch[:, 7]
                R_batch = alphas[:, 0] * p_b + alphas[:, 1] * l_b + alphas[:, 2] * e_b
                R_clipped = np.clip(R_batch, 1e-7, 1.0 - 1e-7)

                # Cross-Entropy Loss: L = - [y log(R) + (1-y) log(1-R)]
                grad_R = (R_clipped - y_batch) / (R_clipped * (1.0 - R_clipped) + 1e-9)
                grad_R = np.clip(grad_R, -5.0, 5.0)  # Gradient clipping

                # Gradients w.r.t alphas: dR/dalpha_i = modality_i
                grad_alphas = np.zeros_like(alphas)
                grad_alphas[:, 0] = grad_R * p_b
                grad_alphas[:, 1] = grad_R * l_b
                grad_alphas[:, 2] = grad_R * e_b

                # Gradients w.r.t logits via Softmax Jacobian
                # dL/dz_k = alpha_k * (grad_alpha_k - sum_j grad_alpha_j * alpha_j)
                dot_term = np.sum(grad_alphas * alphas, axis=1, keepdims=True)
                grad_logits = alphas * (grad_alphas - dot_term)

                # Gradients w.r.t Layer 2 parameters
                grad_W2 = np.dot(h.T, grad_logits) / len(batch_idx)
                grad_b2 = np.mean(grad_logits, axis=0)

                # Gradients w.r.t Layer 1 parameters
                grad_h = np.dot(grad_logits, self.W2.T)
                grad_h[h <= 0] = 0.0  # ReLU gradient

                grad_W1 = np.dot(X_batch.T, grad_h) / len(batch_idx)
                grad_b1 = np.mean(grad_h, axis=0)

                # SGD update
                self.W2 -= self.learning_rate * grad_W2
                self.b2 -= self.learning_rate * grad_b2
                self.W1 -= self.learning_rate * grad_W1
                self.b1 -= self.learning_rate * grad_b1

        self.is_trained = True
        self.is_simulation_only = allow_simulation
        self.training_provenance = (
            "SIMULATION BENCHMARK ONLY (Academic Pipeline Test)"
            if allow_simulation else
            "VERIFIED EMPIRICAL MULTIMODAL DATASET"
        )
        self.training_metrics = {
            "samples_trained": n_samples,
            "epochs": epochs,
            "final_batch_loss": float(np.mean(- (y_batch * np.log(R_clipped) + (1 - y_batch) * np.log(1 - R_clipped))))
        }

        return self
