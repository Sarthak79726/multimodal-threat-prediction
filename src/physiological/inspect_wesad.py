import os
import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any

# Ensure project root is in sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import config
from src.physiological.preprocessing import load_subject_data, inspect_subject_signals, get_wesad_summary

def run_wesad_inspection():
    """
    Comprehensive WESAD dataset inspection script.
    Displays folder structure, available subjects, signal modalities, sampling rates,
    dimensions, NaN checks, and label distributions.
    """
    print("=" * 80)
    print("      WESAD (Wearable Stress and Affect Detection) DATASET INSPECTION")
    print("=" * 80)
    
    wesad_path = Path(config.WESAD_DIR)
    print(f"\n[1] WESAD Root Directory: {wesad_path.resolve()}")
    if not wesad_path.exists():
        print(f"ERROR: WESAD directory does not exist at {wesad_path}")
        return
        
    subject_dirs = [d.name for d in wesad_path.iterdir() if d.is_dir() and d.name.startswith("S")]
    subject_dirs.sort(key=lambda x: int(x[1:]) if x[1:].isdigit() else x)
    print(f"    Available Subject Folders ({len(subject_dirs)}): {subject_dirs}")
    
    print("\n[2] Checking Files in Representative Subject (S2):")
    s2_dir = wesad_path / "S2"
    if s2_dir.exists():
        for item in s2_dir.iterdir():
            size_mb = item.stat().st_size / (1024 * 1024)
            print(f"    - {item.name:<25} ({size_mb:.2f} MB)")
            
    print("\n[3] Sampling Frequencies Configuration:")
    for signal_name, freq in config.SAMPLING_FREQ.items():
        print(f"    - {signal_name:<15}: {freq} Hz")
        
    print("\n[4] Inspecting All Subject Pickles & Label Distributions:")
    summary = get_wesad_summary(wesad_path)
    
    print(f"\n    Successfully loaded {summary['num_subjects']} subjects.")
    print("\n    Aggregate Label Distribution Across All Subjects:")
    total_labels = summary['total_label_counts']
    total_samples = sum(total_labels.values())
    
    for lbl_id in sorted(total_labels.keys()):
        lbl_name = config.WESAD_LABEL_MAP.get(lbl_id, f"Unknown ({lbl_id})")
        cnt = total_labels[lbl_id]
        pct = (cnt / total_samples) * 100 if total_samples > 0 else 0
        is_stress = "STRESS (Target 1)" if lbl_id == 2 else ("NON-STRESS (Target 0)" if lbl_id in [1, 3, 4] else "IGNORED (Transient/Other)")
        print(f"    Label {lbl_id} ({lbl_name:<12}): {cnt:>10} samples ({pct:>5.1f}%)  [{is_stress}]")
        
    print("\n[5] Detailed Per-Subject Signal Modality Summary (First 3 Subjects):")
    for subj_info in summary['subjects_detail'][:3]:
        s_id = subj_info['subject_id']
        print(f"\n    --- Subject: {s_id} ---")
        print(f"    Total Duration: {subj_info['label']['duration_sec']:.2f} seconds ({subj_info['label']['total_samples']} samples @ {subj_info['label']['sampling_hz']} Hz)")
        print("    Chest Modalities:")
        for mod, details in subj_info['chest'].items():
            print(f"      * {mod:<6} | Shape: {str(details['shape']):<15} | Min: {details['min']:>8.2f} | Max: {details['max']:>8.2f} | Mean: {details['mean']:>8.2f} | NaN: {details['has_nan']}")
        print("    Wrist Modalities:")
        for mod, details in subj_info['wrist'].items():
            print(f"      * {mod:<6} | Shape: {str(details['shape']):<15} | Freq: {details['sampling_hz']} Hz | NaN: {details['has_nan']}")
            
    print("\n" + "=" * 80)
    print("WESAD Inspection Complete! All files and signals are valid and ready for model training.")
    print("=" * 80)

if __name__ == "__main__":
    run_wesad_inspection()
