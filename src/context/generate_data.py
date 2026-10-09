import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Ensure project root is in sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
import config

def generate_context_dataset(
    num_samples: int = 2500,
    output_csv: Path = config.DATA_DIR / "context" / "context_data.csv",
    seed: int = 42
) -> pd.DataFrame:
    """
    Generates a synthetic contextual telemetry dataset for Model 3.
    """
    np.random.seed(seed)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    
    movement_speed = np.random.exponential(scale=1.5, size=num_samples) # m/s
    movement_speed = np.clip(movement_speed, 0.0, 15.0)
    
    accel_var = np.random.exponential(scale=1.0, size=num_samples) # m/s^2 variance
    accel_var = np.clip(accel_var, 0.0, 20.0)
    
    solitary = np.random.binomial(n=1, p=0.45, size=num_samples)
    battery = np.random.uniform(2.0, 100.0, size=num_samples)
    ambient_noise = np.random.normal(loc=55.0, scale=15.0, size=num_samples) # dB
    ambient_noise = np.clip(ambient_noise, 25.0, 115.0)
    
    panic_button = np.random.binomial(n=1, p=0.04, size=num_samples) # 4% explicit panic triggers
    hours = np.random.uniform(0.0, 24.0, size=num_samples)
    
    is_late_night = ((hours >= 22) | (hours <= 4)).astype(float)
    is_low_battery = (battery < 15.0).astype(float)
    is_high_speed_running = ((movement_speed > 3.5) & (movement_speed < 8.0)).astype(float) # running away
    
    score = (
        0.80 * panic_button +
        0.35 * (accel_var / 5.0) +
        0.30 * solitary +
        0.25 * is_high_speed_running +
        0.20 * is_late_night +
        0.15 * is_low_battery +
        0.15 * (ambient_noise > 85.0).astype(float) +
        np.random.normal(0, 0.05, size=num_samples)
    )
    
    labels = (score > 0.45).astype(int)
    
    df = pd.DataFrame({
        "movement_speed_mps": np.round(movement_speed, 2),
        "acceleration_variance": np.round(accel_var, 2),
        "solitary_flag": solitary,
        "battery_level_pct": np.round(battery, 1),
        "ambient_noise_db": np.round(ambient_noise, 1),
        "panic_button_pressed": panic_button,
        "time_of_day_hour": np.round(hours, 2),
        "context_risk_label": labels
    })
    
    df.to_csv(output_csv, index=False)
    print(f"Context dataset generated with shape {df.shape} -> Saved to: {output_csv}", flush=True)
    print(f"Target Distribution: Class 0 (Safe): {np.sum(labels==0)}, Class 1 (High Risk): {np.sum(labels==1)}", flush=True)
    return df

if __name__ == "__main__":
    generate_context_dataset()
