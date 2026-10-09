import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Ensure project root is in sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
import config

def generate_environment_dataset(
    num_samples: int = 3000,
    output_csv: Path = config.DATA_DIR / "environment" / "environment_data.csv",
    seed: int = 42
) -> pd.DataFrame:
    """
    Generates a realistic environment context dataset for Model 3.
    Features:
    - time_of_day_hour (0.0 to 24.0)
    - lighting_condition ('Light' or 'Dark')
    - crowd_density ('Low', 'Medium', or 'High')
    
    Target:
    - environment_risk_label (0 = Low Risk, 1 = High Risk)
    """
    np.random.seed(seed)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    
    records = []
    
    for _ in range(num_samples):
        time_hour = float(np.random.uniform(0.0, 24.0))
        is_night = (time_hour >= 19.0 or time_hour <= 5.5)
        
        # Probabilistic lighting based on time
        if is_night:
            lighting = np.random.choice(['Dark', 'Light'], p=[0.8, 0.2])
        else:
            lighting = np.random.choice(['Light', 'Dark'], p=[0.9, 0.1])
            
        # Crowd density based on time & lighting
        if is_night:
            crowd = np.random.choice(['Low', 'Medium', 'High'], p=[0.7, 0.2, 0.1])
        else:
            crowd = np.random.choice(['Low', 'Medium', 'High'], p=[0.15, 0.45, 0.40])
            
        # Graduated time risk
        if time_hour >= 23.0 or time_hour <= 4.5:
            time_risk = 0.35  # Deep late night
        elif (20.0 <= time_hour < 23.0) or (4.5 < time_hour <= 6.0):
            time_risk = 0.22  # Evening / dawn
        else:
            time_risk = 0.05  # Daytime
            
        lighting_risk = 0.35 if lighting == 'Dark' else 0.08
        crowd_risk = 0.32 if crowd == 'Low' else (0.15 if crowd == 'Medium' else 0.02)
        
        noise = np.random.normal(0, 0.03)
        raw_score = np.clip(time_risk + lighting_risk + crowd_risk + noise, 0.05, 1.0)
        label = 1 if raw_score >= 0.65 else 0
        
        records.append({
            "time_of_day_hour": round(time_hour, 2),
            "lighting_condition": lighting,
            "crowd_density": crowd,
            "environment_risk_label": label
        })
        
    df = pd.DataFrame(records)
    df.to_csv(output_csv, index=False)
    
    print(f"Generated Environment Risk Dataset with shape {df.shape} -> Saved to: {output_csv}", flush=True)
    print(f"Class Distribution: Low Risk (0): {sum(df['environment_risk_label']==0)}, High Risk (1): {sum(df['environment_risk_label']==1)}", flush=True)
    return df

if __name__ == "__main__":
    generate_environment_dataset()
