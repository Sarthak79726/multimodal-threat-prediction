import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Ensure project root is in sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
import config
from src.location.campus_map import CampusMapAnalyzer


def generate_location_dataset(
    num_samples: int = 3000,
    output_csv: Path = config.DATA_DIR / "location" / "location_data.csv",
    seed: int = 42
) -> pd.DataFrame:
    """
    Generates a realistic location safety dataset incorporating GPS coordinates
    around BIT Mesra Campus, distance to security outposts, and spatial environmental parameters.
    """
    np.random.seed(seed)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    landmarks = config.BIT_MESRA_LANDMARKS
    analyzer = CampusMapAnalyzer(landmarks)

    records = []

    # Generate samples anchored around BIT Mesra landmarks with spatial noise
    for _ in range(num_samples):
        # Pick a base landmark
        lm = np.random.choice(landmarks)

        # Perturb lat/lon slightly around landmark
        lat_offset = np.random.normal(0, 0.002) # ~200m gaussian std
        lon_offset = np.random.normal(0, 0.002)

        lat = lm["latitude"] + lat_offset
        lon = lm["longitude"] + lon_offset

        time_of_day = float(np.random.uniform(0.0, 24.0))

        # Perform campus map analysis
        res = analyzer.analyze_gps(lat, lon, time_of_day_hour=time_of_day)

        # Ground truth risk calculation
        is_late_night = 1.0 if (time_of_day >= 22.0 or time_of_day <= 4.0) else 0.0

        # Risk increases with crime rate, isolation, distance to security, unlit areas, and late night
        sec_dist_factor = min(1.0, res["distance_to_security_m"] / 700.0)
        lighting_factor = (10.0 - res["lighting_level"]) / 10.0

        risk_score = (
            0.30 * res["crime_rate_index"] +
            0.25 * lighting_factor +
            0.20 * res["isolation_index"] +
            0.15 * sec_dist_factor +
            0.15 * is_late_night +
            np.random.normal(0, 0.04)
        )

        label = 1 if risk_score > 0.42 else 0

        records.append({
            "latitude": round(lat, 5),
            "longitude": round(lon, 5),
            "distance_to_security_m": res["distance_to_security_m"],
            "crime_rate_index": res["crime_rate_index"],
            "lighting_level": res["lighting_level"],
            "isolation_index": res["isolation_index"],
            "time_of_day_hour": round(time_of_day, 2),
            "location_type": res["location_type"],
            "location_risk_label": label
        })

    df = pd.DataFrame(records)
    df.to_csv(output_csv, index=False)

    safe_count = int(np.sum(df["location_risk_label"] == 0))
    risk_count = int(np.sum(df["location_risk_label"] == 1))

    print(f"BIT Mesra GPS location dataset generated with shape {df.shape} -> Saved to: {output_csv}", flush=True)
    print(f"Target Distribution: Class 0 (Safe): {safe_count}, Class 1 (High Risk): {risk_count}", flush=True)
    return df

if __name__ == "__main__":
    generate_location_dataset()

