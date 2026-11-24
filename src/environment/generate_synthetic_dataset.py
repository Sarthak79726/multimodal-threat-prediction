"""
Synthetic Environmental Dataset Generator.
Generates 73,440 records (17 campus locations x 24 hours x 180 simulated days)
based on the official BIT Mesra Campus Crowd and Lighting Schedule baseline.
Chronological splits:
- Train: Days 1 to 120 (48,960 records)
- Validation: Days 121 to 150 (12,240 records)
- Test: Days 151 to 180 (12,240 records)
"""

import sys
import shutil
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import config

LOCATIONS = [
    "Main Building",
    "R&D",
    "H1",
    "H2",
    "H3",
    "H4",
    "Girls Hostel Area / H8-H9",
    "PMC",
    "H6",
    "H7",
    "Dispensary",
    "H10",
    "H11",
    "H12",
    "H13",
    "Sports Complex",
    "Main Gate"
]

def generate_synthetic_environmental_dataset(
    baseline_excel_path: Path = Path("C:/Downloads/BIT_Mesra_Campus_Crowd_Dataset.xlsx"),
    output_dir: Path = config.DATA_DIR / "environment",
    seed: int = 42
) -> pd.DataFrame:
    """
    Generates the 73,440-record synthetic environmental dataset.
    Preserves original schedule and writes both .xlsx and .csv.
    """
    np.random.seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Ensure baseline Excel file is archived in data/environment/
    target_baseline_xlsx = output_dir / "BIT_Mesra_Campus_Crowd_Dataset(1).xlsx"
    if baseline_excel_path.exists() and not target_baseline_xlsx.exists():
        shutil.copyfile(baseline_excel_path, target_baseline_xlsx)
        print(f"[Generator] Preserved baseline schedule to: {target_baseline_xlsx}")

    # Read baseline schedule
    if baseline_excel_path.exists():
        df_baseline = pd.read_excel(baseline_excel_path, sheet_name="Campus_Dataset")
    elif target_baseline_xlsx.exists():
        df_baseline = pd.read_excel(target_baseline_xlsx, sheet_name="Campus_Dataset")
    else:
        raise FileNotFoundError(f"Baseline workbook not found at {baseline_excel_path} or {target_baseline_xlsx}")

    # Map baseline rules by (location, hour)
    baseline_map = {}
    for _, row in df_baseline.iterrows():
        key = (str(row["location"]).strip(), int(row["hour_ist"]))
        baseline_map[key] = {
            "crowd_density": str(row["crowd_density"]).strip(),
            "lighting_condition": str(row["lighting_condition"]).strip()
        }

    # Simulation Parameters
    num_days = 180
    start_date = datetime(2026, 1, 1)
    day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    records = []
    
    for day_idx in range(num_days):
        day_num = day_idx + 1
        curr_date = start_date + timedelta(days=day_idx)
        dow_num = curr_date.weekday()  # 0=Monday, 6=Sunday
        dow_name = day_names[dow_num]
        is_weekend = 1 if dow_num in [5, 6] else 0

        # Determine chronological split
        if day_num <= 120:
            split = "Train"
        elif day_num <= 150:
            split = "Validation"
        else:
            split = "Test"

        # Monthly / seasonal slight fluctuation
        month = curr_date.month

        for loc in LOCATIONS:
            for hour in range(24):
                base = baseline_map.get((loc, hour), {"crowd_density": "Low", "lighting_condition": "Good"})
                base_crowd = base["crowd_density"]
                base_light = base["lighting_condition"]

                # ==========================================
                # 1. SIMULATE CROWD DENSITY CATEGORY
                # ==========================================
                # Logic:
                # - If baseline is "High":
                #   - On weekdays in academic zones (Main Building, R&D): High 90%, Medium 10%
                #   - On weekends in academic zones: drops to Low (60%), Medium (30%), High (10%)
                #   - In hostels on weekends: stays High/Medium
                #   - Sports complex peaks on weekend mornings/evenings (High), drops at night
                #   - Transition hours (6-8, 12-14, 18-20) have higher chance of "Medium"
                # - If baseline is "Low":
                #   - Mostly Low (85%), Medium (13%), High (2%)
                
                is_academic = loc in ["Main Building", "R&D"]
                is_sports = loc == "Sports Complex"
                is_hostel = loc in ["H1", "H2", "H3", "H4", "H6", "H7", "H10", "H11", "H12", "H13", "Girls Hostel Area / H8-H9"]

                if base_crowd == "High":
                    if is_weekend and is_academic:
                        p_crowd = [0.60, 0.30, 0.10]  # [Low, Medium, High]
                    elif hour in [7, 8, 13, 14, 20, 21]:
                        p_crowd = [0.08, 0.42, 0.50]
                    else:
                        p_crowd = [0.05, 0.15, 0.80]
                else:  # base_crowd == "Low"
                    if is_weekend and is_sports and (7 <= hour <= 10 or 16 <= hour <= 19):
                        p_crowd = [0.15, 0.45, 0.40]
                    elif hour in [6, 7, 12, 13, 21, 22]:
                        p_crowd = [0.65, 0.30, 0.05]
                    else:
                        p_crowd = [0.88, 0.10, 0.02]

                sim_crowd = np.random.choice(["Low", "Medium", "High"], p=p_crowd)

                # ==========================================
                # 2. SIMULATE LIGHTING CONDITION CATEGORY
                # ==========================================
                # Logic:
                # Daytime (06:00 to 18:00): Outdoor ambient sunlight makes conditions "Good"
                # (except rare heavy overcast / storms: 2% Low)
                # Nighttime (18:00 to 06:00):
                # - If baseline is "Good" (Main Gate, Dispensary, PMC):
                #   - Continuous lighting: Good (88%), Low (10%), Dark (2% e.g. outage)
                # - If baseline is "Low" (H1-H7, Sports Complex):
                #   - Moderate street lighting: Low (75%), Dark (20%), Good (5%)
                # - If baseline is "Dark" (H10-H13 forest boundary):
                #   - Deep perimeter night: Dark (85%), Low (13%), Good (2%)
                if 6 <= hour < 18:
                    # Daytime
                    p_light = [0.005, 0.035, 0.96]  # [Dark, Low, Good]
                else:
                    # Nighttime
                    if base_light == "Good":
                        p_light = [0.02, 0.12, 0.86]
                    elif base_light == "Low":
                        p_light = [0.22, 0.72, 0.06]
                    else:  # Dark
                        p_light = [0.85, 0.13, 0.02]

                sim_light = np.random.choice(["Dark", "Low", "Good"], p=p_light)

                # ==========================================
                # 3. SIMULATE LIGHTING LUX ESTIMATE (Continuous)
                # ==========================================
                # Scientific boundaries:
                # Dark: 0.1 to 15.0 lux (mean ~4.0, std ~2.5)
                # Low: 15.0 to 120.0 lux (mean ~55.0, std ~18.0)
                # Good:
                #   If daytime (6-17): 300 to 8000 lux (mean ~2500, std ~800)
                #   If nighttime (18-5): 120 to 350 lux (mean ~220, std ~45)
                if sim_light == "Dark":
                    lux = float(np.clip(np.random.normal(4.0, 2.5), 0.1, 15.0))
                elif sim_light == "Low":
                    lux = float(np.clip(np.random.normal(55.0, 18.0), 15.1, 120.0))
                else:  # Good
                    if 6 <= hour < 18:
                        lux = float(np.clip(np.random.normal(2500.0, 800.0), 300.0, 9500.0))
                    else:
                        lux = float(np.clip(np.random.normal(220.0, 45.0), 121.0, 450.0))

                records.append({
                    "simulated_day": day_num,
                    "date": curr_date.strftime("%Y-%m-%d"),
                    "day_of_week": dow_name,
                    "day_of_week_num": dow_num,
                    "weekend_indicator": is_weekend,
                    "hour_of_day": hour,
                    "location": loc,
                    "simulated_crowd_density_category": sim_crowd,
                    "simulated_lighting_category": sim_light,
                    "simulated_lighting_lux_estimate": round(lux, 2),
                    "split": split,
                    "provenance": "SYNTHETIC_SIMULATION_RULE_GENERATED"
                })

    df = pd.DataFrame(records)
    
    # Save CSV
    csv_path = output_dir / "BIT_Mesra_Synthetic_Environmental_Dataset.csv"
    df.to_csv(csv_path, index=False)
    print(f"[Generator] Saved CSV dataset ({len(df)} rows) to: {csv_path}")

    # Save Excel
    xlsx_path = output_dir / "BIT_Mesra_Synthetic_Environmental_Dataset.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Synthetic_Dataset", index=False)
    print(f"[Generator] Saved Excel dataset ({len(df)} rows) to: {xlsx_path}")

    return df

if __name__ == "__main__":
    df_gen = generate_synthetic_environmental_dataset()
    print("Dataset Summary:")
    print(f"Total Rows: {len(df_gen)}")
    print("Split counts:\n", df_gen["split"].value_counts())
    print("Crowd distribution:\n", df_gen["simulated_crowd_density_category"].value_counts(normalize=True))
    print("Lighting distribution:\n", df_gen["simulated_lighting_category"].value_counts(normalize=True))
