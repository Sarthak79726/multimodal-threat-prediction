import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
WESAD_DIR = DATA_DIR / "WESAD"
MODELS_DIR = BASE_DIR / "models"
NOTEBOOKS_DIR = BASE_DIR / "notebooks"

# Subject configuration
ALL_SUBJECTS = ["S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10", "S11", "S13", "S14", "S15", "S16", "S17"]

# WESAD Signal Sampling Frequencies (Hz)
SAMPLING_FREQ = {
    "chest": 700,
    "wrist_ACC": 32,
    "wrist_BVP": 64,
    "wrist_EDA": 4,
    "wrist_TEMP": 4,
}

# Original WESAD Label Mapping
# 0 = transient/undefined, 1 = baseline, 2 = stress, 3 = amusement, 4 = meditation
WESAD_LABEL_MAP = {
    0: "transient",
    1: "baseline",
    2: "stress",
    3: "amusement",
    4: "meditation"
}

# Binary Classification Mapping for Initial Physiological Model:
# 0 = Non-Stress (baseline: 1, amusement: 3, meditation: 4)
# 1 = Stress (stress: 2)
# Ignores label 0 (transient) and >4
BINARY_LABEL_MAP = {
    1: 0, # Baseline -> Non-Stress
    2: 1, # Stress -> Stress
    3: 0, # Amusement -> Non-Stress
    4: 0, # Meditation -> Non-Stress
}

# Risk Level Classification Thresholds (Configurable)
RISK_THRESHOLDS = {
    "LOW_MAX": 0.39,
    "MEDIUM_MAX": 0.69,
    # > 0.69 is HIGH
}

# BIT Mesra Campus GPS Configuration & Centralized Landmarks Database
from src.location.landmarks import BIT_MESRA_CENTER, BIT_MESRA_LANDMARKS_DB, get_all_landmarks_list

BIT_MESRA_LANDMARKS = get_all_landmarks_list()

def get_risk_level(probability: float) -> str:
    """Classifies risk probability into LOW, MEDIUM, or HIGH."""
    if probability <= RISK_THRESHOLDS["LOW_MAX"]:
        return "LOW"
    elif probability <= RISK_THRESHOLDS["MEDIUM_MAX"]:
        return "MEDIUM"
    else:
        return "HIGH"

# BIT Mesra Campus Timezone Configuration (Ranchi, Jharkhand, India - IST: UTC+5:30)
try:
    from zoneinfo import ZoneInfo
    CAMPUS_TIMEZONE = ZoneInfo("Asia/Kolkata")
except Exception:
    import datetime
    CAMPUS_TIMEZONE = datetime.timezone(datetime.timedelta(hours=5, minutes=30))

def get_campus_now():
    """Returns the current datetime localized to BIT Mesra Campus (Asia/Kolkata / IST) timezone."""
    import datetime
    return datetime.datetime.now(CAMPUS_TIMEZONE)


