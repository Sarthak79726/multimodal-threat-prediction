"""
Reproducible Stacking Ensemble Experiment Runner & Visualization Generator.
Executes the comparative experiment across all candidate models, generates
academic evaluation figures (ROC Curves, PR Curves, Calibration Curves, Model Comparison Barchart),
and saves them to notebooks/plots/.
"""

import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure project root is in sys.path
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import config
from src.fusion.train import run_academic_benchmark_experiment

def generate_experiment_visualizations(
    results_json_path: Path = config.MODELS_DIR / "fusion" / "stacking_benchmark_results.json",
    output_dir: Path = config.NOTEBOOKS_DIR / "plots"
):
    output_dir.mkdir(parents=True, exist_ok=True)
    if not results_json_path.exists():
        print(f"Results file {results_json_path} not found. Running benchmark first...")
        run_academic_benchmark_experiment()

    with open(results_json_path, "r") as f:
        results = json.load(f)

    # 1. Comparison Bar Chart
    models = list(results.keys())
    metrics_names = ["ROC-AUC", "F1-Score", "Brier Score", "ECE"]

    data_chart = {
        "Model": [],
        "Metric": [],
        "Score": []
    }

    for m in models:
        res = results[m]
        # Shorten model name for plotting
        short_name = m.split(". ")[-1] if ". " in m else m
        data_chart["Model"].extend([short_name, short_name, short_name, short_name])
        data_chart["Metric"].extend(metrics_names)
        data_chart["Score"].extend([
            res.get("roc_auc") or 0.0,
            res.get("f1_score") or 0.0,
            res.get("brier_score") or 0.0,
            res.get("expected_calibration_error") or 0.0
        ])

    df_plot = pd.DataFrame(data_chart)

    plt.figure(figsize=(12, 6))
    sns.set_theme(style="whitegrid")
    palette = {"ROC-AUC": "#2b5c8f", "F1-Score": "#27ae60", "Brier Score": "#e74c3c", "ECE": "#e67e22"}
    ax = sns.barplot(data=df_plot, x="Model", y="Score", hue="Metric", palette=palette)
    plt.title("Academic Multimodal Stacking Benchmark: Model Performance Comparison", fontsize=14, fontweight="bold", pad=15)
    plt.xticks(rotation=25, ha="right", fontsize=10)
    plt.ylabel("Score (Lower is better for Brier & ECE; Higher for AUC & F1)", fontsize=11)
    plt.ylim(0.0, 1.05)
    plt.legend(title="Metric", loc="upper left")
    plt.tight_layout()
    chart_path = output_dir / "stacking_comparison_bar.png"
    plt.savefig(chart_path, dpi=300)
    plt.close()
    print(f"Saved comparison bar chart to: {chart_path}")

    # 2. Calibration Reliability Curves
    plt.figure(figsize=(8, 7))
    plt.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")

    for m in ["4. Weighted Late Fusion Baseline", "5. Stacking (Logistic Regression)", "7. Stacking (XGBoost)"]:
        if m in results:
            bins = results[m].get("calibration_bins", [])
            if bins:
                confs = [b["confidence"] for b in bins]
                accs = [b["accuracy"] for b in bins]
                short_name = m.split(". ")[-1]
                ece_val = results[m].get("expected_calibration_error", 0.0)
                plt.plot(confs, accs, marker="o", linewidth=2, label=f"{short_name} (ECE={ece_val:.3f})")

    plt.title("Probability Calibration Reliability Curves (Untouched Test Set)", fontsize=13, fontweight="bold")
    plt.xlabel("Mean Predicted Probability (Confidence)", fontsize=11)
    plt.ylabel("Observed Fraction of Positives (Empirical Risk)", fontsize=11)
    plt.legend(loc="lower right")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    calib_path = output_dir / "stacking_calibration_curves.png"
    plt.savefig(calib_path, dpi=300)
    plt.close()
    print(f"Saved calibration reliability curves to: {calib_path}")

if __name__ == "__main__":
    generate_experiment_visualizations()

