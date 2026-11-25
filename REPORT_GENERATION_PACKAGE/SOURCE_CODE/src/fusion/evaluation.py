"""
Model Evaluation and Experimental Comparison Suite.
Calculates comprehensive, academically rigorous evaluation metrics:
- Accuracy, Precision, Recall, F1-Score
- ROC-AUC, PR-AUC (Average Precision)
- Brier Score Loss (probabilistic calibration measure)
- Expected Calibration Error (ECE) & Reliability Bins
- Confusion Matrix
- Multi-Model Benchmarking (Base vs Baseline vs Stacking architectures)
- Serialization to JSON & CSV reports.
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, brier_score_loss,
    confusion_matrix
)

# Ensure project root is in sys.path
_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import config

def compute_expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10
) -> Tuple[float, List[Dict[str, float]]]:
    """
    Computes Expected Calibration Error (ECE) across uniform confidence bins.
    
    ECE = sum_{b=1}^B (|B_b| / N) * |acc(B_b) - conf(B_b)|
    """
    y_true_arr = np.asarray(y_true, dtype=np.int32)
    y_prob_arr = np.asarray(y_prob, dtype=np.float64)
    n_total = len(y_true_arr)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    bin_details = []

    for i in range(n_bins):
        bin_lower = bins[i]
        bin_upper = bins[i + 1]
        
        if i == n_bins - 1:
            in_bin = (y_prob_arr >= bin_lower) & (y_prob_arr <= bin_upper)
        else:
            in_bin = (y_prob_arr >= bin_lower) & (y_prob_arr < bin_upper)

        bin_size = int(np.sum(in_bin))
        if bin_size > 0:
            bin_acc = float(np.mean(y_true_arr[in_bin]))
            bin_conf = float(np.mean(y_prob_arr[in_bin]))
            bin_error = abs(bin_acc - bin_conf)
            ece += (bin_size / n_total) * bin_error

            bin_details.append({
                "bin_idx": i,
                "bin_range": [round(bin_lower, 2), round(bin_upper, 2)],
                "bin_size": bin_size,
                "accuracy": round(bin_acc, 4),
                "confidence": round(bin_conf, 4),
                "error": round(bin_error, 4)
            })

    return round(float(ece), 4), bin_details

def evaluate_predictions(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.50,
    model_name: str = "Model"
) -> Dict[str, Any]:
    """
    Computes complete statistical and probabilistic metrics for a binary classifier.
    """
    y_t = np.asarray(y_true, dtype=np.int32)
    y_p = np.asarray(y_prob, dtype=np.float64)
    y_pred = (y_p >= threshold).astype(int)

    acc = float(accuracy_score(y_t, y_pred))
    prec = float(precision_score(y_t, y_pred, zero_division=0))
    rec = float(recall_score(y_t, y_pred, zero_division=0))
    f1 = float(f1_score(y_t, y_pred, zero_division=0))

    # ROC-AUC and PR-AUC require >1 unique class
    if len(np.unique(y_t)) > 1:
        roc_auc = float(roc_auc_score(y_t, y_p))
        pr_auc = float(average_precision_score(y_t, y_p))
    else:
        roc_auc = float("nan")
        pr_auc = float("nan")

    brier = float(brier_score_loss(y_t, y_p))
    cm = confusion_matrix(y_t, y_pred).tolist()
    ece, bin_details = compute_expected_calibration_error(y_t, y_p, n_bins=10)

    # Class balance summary
    support = {
        "class_0_count": int(np.sum(y_t == 0)),
        "class_1_count": int(np.sum(y_t == 1)),
        "total_samples": len(y_t)
    }

    return {
        "model_name": model_name,
        "threshold": threshold,
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(roc_auc, 4) if not np.isnan(roc_auc) else None,
        "pr_auc": round(pr_auc, 4) if not np.isnan(pr_auc) else None,
        "brier_score": round(brier, 4),
        "expected_calibration_error": ece,
        "confusion_matrix": cm,
        "support": support,
        "calibration_bins": bin_details
    }

def format_comparison_table(results_dict: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """
    Formats dictionary of model results into a structured DataFrame.
    """
    rows = []
    for model_name, metrics in results_dict.items():
        rows.append({
            "Model / Branch": model_name,
            "Accuracy": metrics.get("accuracy"),
            "Precision": metrics.get("precision"),
            "Recall": metrics.get("recall"),
            "F1-Score": metrics.get("f1_score"),
            "ROC-AUC": metrics.get("roc_auc"),
            "PR-AUC": metrics.get("pr_auc"),
            "Brier Score": metrics.get("brier_score"),
            "ECE": metrics.get("expected_calibration_error")
        })
    return pd.DataFrame(rows)

def save_evaluation_report(
    results_dict: Dict[str, Dict[str, Any]],
    output_dir: Path = config.MODELS_DIR / "fusion",
    filename_prefix: str = "stacking_evaluation_report"
) -> Tuple[Path, Path]:
    """
    Saves evaluation comparison as JSON and CSV reports.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{filename_prefix}.json"
    csv_path = output_dir / f"{filename_prefix}.csv"

    # Save JSON report
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results_dict, f, indent=2)

    # Save CSV comparison table
    df_cmp = format_comparison_table(results_dict)
    df_cmp.to_csv(csv_path, index=False)

    print(f"[Evaluation] Saved evaluation report to:\n  - {json_path}\n  - {csv_path}", flush=True)
    return json_path, csv_path

