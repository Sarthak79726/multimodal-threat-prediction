import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Ensure project root is in sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
import config

def generate_fusion_dataset(
    num_samples: int = 4000,
    output_csv: Path = config.DATA_DIR / "fusion" / "fusion_data.csv",
    seed: int = 42
) -> pd.DataFrame:
    """
    Generates a dataset of sub-model risk probabilities:
    [physiological_stress_score, location_risk_score, environment_risk_score]
    mapped to ground truth overall threat label (0 = SAFE, 1 = RISKY).
    """
    np.random.seed(seed)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    
    # Generate sub-model risk scores (0.0 to 1.0)
    physiological_stress_score = np.random.beta(a=0.8, b=2.2, size=num_samples)
    location_risk_score = np.random.beta(a=1.0, b=2.0, size=num_samples)
    environment_risk_score = np.random.beta(a=0.8, b=2.5, size=num_samples)
    
    # Ground truth threat function using explainable prototype weights:
    # 40% Physiological, 25% Location, 35% Environment
    weighted_sum = (
        0.40 * physiological_stress_score +
        0.25 * location_risk_score +
        0.35 * environment_risk_score
    )
    
    latent_threat_score = weighted_sum + np.random.normal(0, 0.01, size=num_samples)
    
    # Binary Threat Label (0 = SAFE, 1 = RISKY at 0.50 threshold)
    threat_label = (latent_threat_score >= 0.50).astype(int)
    
    df = pd.DataFrame({
        "physiological_stress_score": np.round(physiological_stress_score, 4),
        "location_risk_score": np.round(location_risk_score, 4),
        "environment_risk_score": np.round(environment_risk_score, 4),
        "threat_label": threat_label
    })
    
    df.to_csv(output_csv, index=False)
    print(f"Fusion dataset generated with shape {df.shape} -> Saved to: {output_csv}", flush=True)
    print(f"Target Distribution: Class 0 (SAFE): {np.sum(threat_label==0)}, Class 1 (RISKY): {np.sum(threat_label==1)}", flush=True)
    return df

if __name__ == "__main__":
    generate_fusion_dataset()
