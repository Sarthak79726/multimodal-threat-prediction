import os
import sys
import numpy as np
import pandas as pd
from scipy import signal, stats
from pathlib import Path
from typing import Dict, List, Tuple, Any

# Ensure project root is in sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import config
from src.physiological.preprocessing import load_subject_data

def compute_signal_features(arr: np.ndarray, prefix: str) -> Dict[str, float]:
    """
    Computes generic statistical features for 1D or multi-D arrays.
    """
    feats = {}
    if arr.ndim == 1 or (arr.ndim == 2 and arr.shape[1] == 1):
        vec = arr.ravel()
        feats[f"{prefix}_mean"] = float(np.mean(vec))
        feats[f"{prefix}_std"] = float(np.std(vec))
        feats[f"{prefix}_min"] = float(np.min(vec))
        feats[f"{prefix}_max"] = float(np.max(vec))
        feats[f"{prefix}_skew"] = float(stats.skew(vec)) if len(vec) > 2 else 0.0
        # Slope/Trend via linear fit
        if len(vec) > 1:
            t = np.arange(len(vec))
            slope, _ = np.polyfit(t, vec, 1)
            feats[f"{prefix}_slope"] = float(slope)
        else:
            feats[f"{prefix}_slope"] = 0.0
    elif arr.ndim == 2 and arr.shape[1] == 3: # 3-axis Accelerometer
        mag = np.linalg.norm(arr, axis=1)
        feats[f"{prefix}_mag_mean"] = float(np.mean(mag))
        feats[f"{prefix}_mag_std"] = float(np.std(mag))
        feats[f"{prefix}_mag_max"] = float(np.max(mag))
        for i, axis_name in enumerate(['x', 'y', 'z']):
            vec = arr[:, i]
            feats[f"{prefix}_{axis_name}_mean"] = float(np.mean(vec))
            feats[f"{prefix}_{axis_name}_std"] = float(np.std(vec))
    return feats

def compute_ecg_bvp_features(arr: np.ndarray, fs: int, prefix: str) -> Dict[str, float]:
    """
    Extracts heart-rate and peak-related features from ECG or BVP signals.
    """
    feats = {}
    vec = arr.ravel()
    
    # Simple peak detection for inter-beat intervals
    # Min distance corresponding to max ~200 BPM -> (fs / (200/60)) = fs * 0.3 sec
    min_dist = int(fs * 0.3)
    peaks, _ = signal.find_peaks(vec, distance=min_dist, prominence=np.std(vec) * 0.5)
    
    feats[f"{prefix}_peak_count"] = float(len(peaks))
    
    if len(peaks) > 1:
        rr_intervals = np.diff(peaks) / fs  # in seconds
        feats[f"{prefix}_hr_bpm"] = float(60.0 / np.mean(rr_intervals))
        feats[f"{prefix}_rr_mean"] = float(np.mean(rr_intervals))
        feats[f"{prefix}_rr_std"] = float(np.std(rr_intervals))
        # RMSSD (Root Mean Square of Successive Differences)
        if len(rr_intervals) > 1:
            rmssd = np.sqrt(np.mean(np.square(np.diff(rr_intervals))))
            feats[f"{prefix}_rmssd"] = float(rmssd)
        else:
            feats[f"{prefix}_rmssd"] = 0.0
    else:
        feats[f"{prefix}_hr_bpm"] = 0.0
        feats[f"{prefix}_rr_mean"] = 0.0
        feats[f"{prefix}_rr_std"] = 0.0
        feats[f"{prefix}_rmssd"] = 0.0
        
    return feats

def extract_window_features(
    subject_id: str,
    chest_signals: Dict[str, np.ndarray],
    wrist_signals: Dict[str, np.ndarray],
    labels: np.ndarray,
    start_sec: float,
    window_sec: float = 60.0
) -> Tuple[Dict[str, Any] | None, int | None]:
    """
    Extracts features across chest and wrist modalities for a specific time window.
    Determines window ground truth label if dominant condition exists.
    """
    fs_chest = config.SAMPLING_FREQ['chest']
    c_start = int(start_sec * fs_chest)
    c_end = int((start_sec + window_sec) * fs_chest)
    
    if c_end > len(labels):
        return None, None
        
    window_labels = labels[c_start:c_end]
    
    # Map raw labels to binary classification targets (1->0 baseline, 2->1 stress, 3->0 amusement, 4->0 meditation)
    mapped_labels = [config.BINARY_LABEL_MAP[lbl] for lbl in window_labels if lbl in config.BINARY_LABEL_MAP]
    
    if len(mapped_labels) < (len(window_labels) * 0.7):
        # Window contains mostly transient (0) or transition (5,6,7) labels
        return None, None
        
    # Assign majority binary label
    counts = np.bincount(mapped_labels)
    target_label = int(np.argmax(counts))
    
    row = {
        "subject_id": subject_id,
        "start_sec": start_sec,
        "label": target_label
    }
    
    # 1. Chest Signals (700 Hz)
    for mod in ['ECG', 'EDA', 'EMG', 'Temp', 'Resp', 'ACC']:
        if mod in chest_signals:
            arr = chest_signals[mod][c_start:c_end]
            row.update(compute_signal_features(arr, f"chest_{mod.lower()}"))
            if mod == 'ECG':
                row.update(compute_ecg_bvp_features(arr, fs_chest, "chest_ecg"))
                
    # 2. Wrist Signals (Multi-rate)
    for mod, fs in [('ACC', 32), ('BVP', 64), ('EDA', 4), ('TEMP', 4)]:
        if mod in wrist_signals:
            w_start = int(start_sec * fs)
            w_end = int((start_sec + window_sec) * fs)
            arr = wrist_signals[mod][w_start:w_end]
            if len(arr) > 0:
                row.update(compute_signal_features(arr, f"wrist_{mod.lower()}"))
                if mod == 'BVP':
                    row.update(compute_ecg_bvp_features(arr, fs, "wrist_bvp"))
                    
    return row, target_label

def extract_dataset_features(
    window_sec: float = 60.0,
    stride_sec: float = 10.0,
    output_path: Path = config.DATA_DIR / "physiological" / "physiological_features.csv"
) -> pd.DataFrame:
    """
    Extracts features for all 15 WESAD subjects using a sliding window.
    Saves the extracted dataset as CSV.
    """
    all_rows = []
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    for subj in config.ALL_SUBJECTS:
        print(f"Extracting features for Subject {subj}...", flush=True)
        data = load_subject_data(subj)
        chest_signals = data['signal']['chest']
        wrist_signals = data['signal']['wrist']
        labels = data['label']
        
        total_duration_sec = len(labels) / config.SAMPLING_FREQ['chest']
        start_sec = 0.0
        
        subj_count = 0
        while (start_sec + window_sec) <= total_duration_sec:
            row, lbl = extract_window_features(
                subject_id=subj,
                chest_signals=chest_signals,
                wrist_signals=wrist_signals,
                labels=labels,
                start_sec=start_sec,
                window_sec=window_sec
            )
            if row is not None:
                all_rows.append(row)
                subj_count += 1
                
            start_sec += stride_sec
            
        print(f"  -> Subject {subj}: Extracted {subj_count} valid window samples.", flush=True)
        
    df = pd.DataFrame(all_rows)
    df.to_csv(output_path, index=False)
    print(f"\nFeature extraction complete! Saved dataset with shape {df.shape} to: {output_path}", flush=True)
    return df

if __name__ == "__main__":
    extract_dataset_features()
