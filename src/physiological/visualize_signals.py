import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any

# Ensure project root is in sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import config
from src.physiological.preprocessing import load_subject_data

def generate_exploratory_summary(subject_id: str = "S2", output_dir: Path = config.DATA_DIR / "physiological"):
    """
    Extracts representative signal windows for Baseline, Stress, and Amusement
    from a subject and generates summary statistics & visualization plots.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Generating exploratory visualizations for Subject {subject_id}...", flush=True)
    
    data = load_subject_data(subject_id)
    chest = data['signal']['chest']
    wrist = data['signal']['wrist']
    labels = data['label']
    
    fs_chest = config.SAMPLING_FREQ['chest']
    
    # Identify sample indices for conditions
    idx_baseline = np.where(labels == 1)[0]
    idx_stress = np.where(labels == 2)[0]
    idx_amusement = np.where(labels == 3)[0]
    
    summary_rows = []
    
    conditions = [
        ("Baseline (Label 1)", idx_baseline),
        ("Stress (Label 2)", idx_stress),
        ("Amusement (Label 3)", idx_amusement)
    ]
    
    for cond_name, idx_arr in conditions:
        if len(idx_arr) == 0:
            continue
            
        # Select middle 60 seconds slice
        mid = len(idx_arr) // 2
        start_idx = idx_arr[max(0, mid - 30 * fs_chest)]
        end_idx = start_idx + 60 * fs_chest
        
        c_ecg = chest['ECG'][start_idx:end_idx].ravel()
        c_eda = chest['EDA'][start_idx:end_idx].ravel()
        c_emg = chest['EMG'][start_idx:end_idx].ravel()
        c_temp = chest['Temp'][start_idx:end_idx].ravel()
        c_resp = chest['Resp'][start_idx:end_idx].ravel()
        
        summary_rows.append({
            "subject": subject_id,
            "condition": cond_name,
            "ecg_mean": float(np.mean(c_ecg)),
            "ecg_std": float(np.std(c_ecg)),
            "eda_mean": float(np.mean(c_eda)),
            "eda_std": float(np.std(c_eda)),
            "emg_mean": float(np.mean(c_emg)),
            "temp_mean": float(np.mean(c_temp)),
            "resp_std": float(np.std(c_resp))
        })
        
    df_summary = pd.DataFrame(summary_rows)
    summary_csv = output_dir / "wesad_condition_summary.csv"
    df_summary.to_csv(summary_csv, index=False)
    print(f"Saved signal summary to: {summary_csv}", flush=True)
    
    # Plot representative ECG & EDA signals under Baseline vs Stress
    plt.figure(figsize=(12, 6))
    
    plt.subplot(2, 1, 1)
    if len(idx_baseline) > 0 and len(idx_stress) > 0:
        b_slice = chest['ECG'][idx_baseline[0]:idx_baseline[0] + 5 * fs_chest].ravel()
        s_slice = chest['ECG'][idx_stress[0]:idx_stress[0] + 5 * fs_chest].ravel()
        t = np.linspace(0, 5, len(b_slice))
        plt.plot(t, b_slice, label='Baseline ECG', alpha=0.8, color='blue')
        plt.plot(t, s_slice, label='Stress ECG', alpha=0.8, color='red')
        plt.title(f"WESAD {subject_id}: 5-Second Chest ECG Signal Comparison (Baseline vs Stress)")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Voltage (mV)")
        plt.legend(loc='upper right')
        plt.grid(True, linestyle='--', alpha=0.5)
        
    plt.subplot(2, 1, 2)
    if len(idx_baseline) > 0 and len(idx_stress) > 0:
        b_eda = chest['EDA'][idx_baseline[0]:idx_baseline[0] + 60 * fs_chest].ravel()
        s_eda = chest['EDA'][idx_stress[0]:idx_stress[0] + 60 * fs_chest].ravel()
        t60 = np.linspace(0, 60, len(b_eda))
        plt.plot(t60, b_eda, label='Baseline EDA', color='blue')
        plt.plot(t60, s_eda, label='Stress EDA', color='red')
        plt.title(f"WESAD {subject_id}: 60-Second Electrodermal Activity (EDA / Skin Conductance)")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Conductance (μS)")
        plt.legend(loc='upper right')
        plt.grid(True, linestyle='--', alpha=0.5)
        
    plt.tight_layout()
    plot_path = output_dir / "wesad_signals_plot.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved exploratory signal plot to: {plot_path}", flush=True)

if __name__ == "__main__":
    generate_exploratory_summary()
