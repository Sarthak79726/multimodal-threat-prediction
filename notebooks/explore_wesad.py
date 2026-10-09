import os
import sys
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.physiological.preprocessing import load_subject_data, inspect_subject_signals, get_wesad_summary
import config

def explore_and_generate_plots():
    print("=== WESAD Dataset Exploration ===", flush=True)
    
    # 1. Scan dataset
    summary = get_wesad_summary()
    print(f"Total Subjects Detected: {summary['num_subjects']}", flush=True)
    print(f"Subject List: {summary['subjects']}", flush=True)
    print("\nTotal Label Distribution Across All Subjects (samples @ 700Hz):", flush=True)
    for lbl, count in sorted(summary['total_label_counts'].items()):
        label_name = config.WESAD_LABEL_MAP.get(lbl, f"Other ({lbl})")
        print(f"  Label {lbl} [{label_name}]: {count:,} samples ({count / 700 / 60:.2f} minutes)", flush=True)

        
    # 2. Detailed inspection for Representative Subject S2
    print("\n--- Inspecting Subject S2 ---")
    s2_raw = load_subject_data("S2")
    s2_info = inspect_subject_signals(s2_raw)
    
    print("\nS2 Chest Signals (Sampling Freq: 700 Hz):")
    for mod, details in s2_info["chest"].items():
        print(f"  {mod:5s} | Shape: {str(details['shape']):15s} | NaN: {details['has_nan']} | Range: [{details['min']:.2f}, {details['max']:.2f}]")
        
    print("\nS2 Wrist Signals:")
    for mod, details in s2_info["wrist"].items():
        print(f"  {mod:5s} | Freq: {details['sampling_hz']} Hz | Shape: {str(details['shape']):15s} | NaN: {details['has_nan']} | Range: [{details['min']:.2f}, {details['max']:.2f}]")
        
    # 3. Plot representative waveforms for S2
    print("\nGenerating representative signal plots for S2...")
    output_dir = Path(config.BASE_DIR) / "notebooks" / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    signals = s2_raw['signal']
    labels = s2_raw['label']
    
    # Select a 10-second snippet (7000 samples @ 700 Hz) during Baseline (label 1) and Stress (label 2)
    baseline_indices = np.where(labels == 1)[0]
    stress_indices = np.where(labels == 2)[0]
    
    b_start = baseline_indices[1000] if len(baseline_indices) > 1000 else 0
    s_start = stress_indices[1000] if len(stress_indices) > 1000 else 0
    
    fs_chest = config.SAMPLING_FREQ['chest']
    snippet_duration = 10  # 10 seconds
    num_samples_chest = snippet_duration * fs_chest
    
    t_chest = np.linspace(0, snippet_duration, num_samples_chest)
    
    fig, axes = plt.subplots(5, 2, figsize=(15, 12), sharex=True)
    fig.suptitle("Subject S2 Representative Physiological Modalities: Baseline vs Stress", fontsize=14, fontweight='bold')
    
    # ECG
    axes[0, 0].plot(t_chest, signals['chest']['ECG'][b_start:b_start+num_samples_chest], color='#1f77b4')
    axes[0, 0].set_title("Baseline: ECG (Chest 700Hz)")
    axes[0, 0].set_ylabel("mV")
    
    axes[0, 1].plot(t_chest, signals['chest']['ECG'][s_start:s_start+num_samples_chest], color='#d62728')
    axes[0, 1].set_title("Stress: ECG (Chest 700Hz)")
    axes[0, 1].set_ylabel("mV")
    
    # EDA (Chest)
    axes[1, 0].plot(t_chest, signals['chest']['EDA'][b_start:b_start+num_samples_chest], color='#2ca02c')
    axes[1, 0].set_title("Baseline: EDA (Chest 700Hz)")
    axes[1, 0].set_ylabel("uS")
    
    axes[1, 1].plot(t_chest, signals['chest']['EDA'][s_start:s_start+num_samples_chest], color='#ff7f0e')
    axes[1, 1].set_title("Stress: EDA (Chest 700Hz)")
    axes[1, 1].set_ylabel("uS")
    
    # Respiration
    axes[2, 0].plot(t_chest, signals['chest']['Resp'][b_start:b_start+num_samples_chest], color='#9467bd')
    axes[2, 0].set_title("Baseline: Respiration (Chest 700Hz)")
    axes[2, 0].set_ylabel("% Signal")
    
    axes[2, 1].plot(t_chest, signals['chest']['Resp'][s_start:s_start+num_samples_chest], color='#8c564b')
    axes[2, 1].set_title("Stress: Respiration (Chest 700Hz)")
    axes[2, 1].set_ylabel("% Signal")
    
    # EMG
    axes[3, 0].plot(t_chest, signals['chest']['EMG'][b_start:b_start+num_samples_chest], color='#7f7f7f')
    axes[3, 0].set_title("Baseline: EMG (Chest 700Hz)")
    axes[3, 0].set_ylabel("mV")
    
    axes[3, 1].plot(t_chest, signals['chest']['EMG'][s_start:s_start+num_samples_chest], color='#bcbd22')
    axes[3, 1].set_title("Stress: EMG (Chest 700Hz)")
    axes[3, 1].set_ylabel("mV")
    
    # Wrist BVP (64Hz -> 640 samples for 10 sec)
    b_start_wrist_bvp = int(b_start * (64 / 700))
    s_start_wrist_bvp = int(s_start * (64 / 700))
    t_wrist_bvp = np.linspace(0, snippet_duration, 640)
    
    axes[4, 0].plot(t_wrist_bvp, signals['wrist']['BVP'][b_start_wrist_bvp:b_start_wrist_bvp+640], color='#17becf')
    axes[4, 0].set_title("Baseline: BVP (Wrist 64Hz)")
    axes[4, 0].set_xlabel("Time (seconds)")
    
    axes[4, 1].plot(t_wrist_bvp, signals['wrist']['BVP'][s_start_wrist_bvp:s_start_wrist_bvp+640], color='#e377c2')
    axes[4, 1].set_title("Stress: BVP (Wrist 64Hz)")
    axes[4, 1].set_xlabel("Time (seconds)")
    
    plt.tight_layout()
    plot_path = output_dir / "wesad_s2_signals.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Plot saved to: {plot_path}")

    # Save summary stats JSON for documentation
    summary_file = output_dir / "wesad_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Summary JSON saved to: {summary_file}")

if __name__ == "__main__":
    explore_and_generate_plots()
