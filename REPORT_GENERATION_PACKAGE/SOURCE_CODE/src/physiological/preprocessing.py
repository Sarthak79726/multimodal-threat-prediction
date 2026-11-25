import os
import pickle
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List
from pathlib import Path
import config

def load_subject_data(subject_id: str, data_dir: Path = config.WESAD_DIR) -> Dict[str, Any]:
    """
    Safely loads raw WESAD pickle file for a given subject ID.
    
    Args:
        subject_id: Identifier string (e.g. 'S2')
        data_dir: Path to WESAD root folder
        
    Returns:
        Dictionary containing raw signal data, labels, and metadata.
    """
    pkl_file = Path(data_dir) / subject_id / f"{subject_id}.pkl"
    if not pkl_file.exists():
        raise FileNotFoundError(f"Pickle file not found at: {pkl_file}")
        
    with open(pkl_file, 'rb') as f:
        # WESAD pickles were created with Python 2, requiring latin1 encoding
        subject_data = pickle.load(f, encoding='latin1')
        
    return subject_data

def inspect_subject_signals(subject_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Inspects array shapes, sampling frequencies, missing values, and data types
    for chest, wrist, and label arrays in a subject data dictionary.
    """
    subject_id = subject_data.get('subject', 'Unknown')
    signals = subject_data.get('signal', {})
    labels = subject_data.get('label', np.array([]))
    
    summary = {
        "subject_id": subject_id,
        "chest": {},
        "wrist": {},
        "label": {
            "total_samples": len(labels),
            "sampling_hz": config.SAMPLING_FREQ["chest"],
            "duration_sec": len(labels) / config.SAMPLING_FREQ["chest"],
            "has_nan": bool(np.isnan(labels).any()),
            "unique_labels": np.unique(labels).tolist(),
            "label_counts": {int(k): int(v) for k, v in zip(*np.unique(labels, return_counts=True))}
        }
    }
    
    # Inspect Chest Modalities
    if 'chest' in signals:
        for mod, arr in signals['chest'].items():
            summary["chest"][mod] = {
                "shape": list(arr.shape),
                "dtype": str(arr.dtype),
                "has_nan": bool(np.isnan(arr).any()),
                "sampling_hz": config.SAMPLING_FREQ["chest"],
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr))
            }
            
    # Inspect Wrist Modalities
    if 'wrist' in signals:
        for mod, arr in signals['wrist'].items():
            freq_key = f"wrist_{mod}"
            hz = config.SAMPLING_FREQ.get(freq_key, "Unknown")
            summary["wrist"][mod] = {
                "shape": list(arr.shape),
                "dtype": str(arr.dtype),
                "has_nan": bool(np.isnan(arr).any()),
                "sampling_hz": hz,
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr))
            }
            
    return summary

def get_wesad_summary(data_dir: Path = config.WESAD_DIR) -> Dict[str, Any]:
    """
    Scans all available WESAD subjects and returns aggregate dataset statistics.
    """
    subjects = config.ALL_SUBJECTS
    dataset_info = []
    total_samples_by_label = {}
    
    for subj in subjects:
        try:
            print(f"Scanning subject {subj}...", flush=True)
            subj_raw = load_subject_data(subj, data_dir)
            subj_info = inspect_subject_signals(subj_raw)
            dataset_info.append(subj_info)
            
            # Accumulate label distributions
            for lbl, cnt in subj_info["label"]["label_counts"].items():
                total_samples_by_label[lbl] = total_samples_by_label.get(lbl, 0) + cnt
        except Exception as e:
            print(f"Error inspecting {subj}: {e}")
            
    return {
        "num_subjects": len(dataset_info),
        "subjects": [s["subject_id"] for s in dataset_info],
        "total_label_counts": total_samples_by_label,
        "subjects_detail": dataset_info
    }
