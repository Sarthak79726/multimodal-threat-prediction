import os
import datetime
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
WESAD_DIR = DATA_DIR / "WESAD"
MODELS_DIR = BASE_DIR / "models"
NOTEBOOKS_DIR = BASE_DIR / "notebooks"

# BIT Mesra Campus Timezone Configuration (Ranchi, Jharkhand, India - IST: UTC+5:30)
# Defined early to prevent circular import AttributeError across modules
try:
    from zoneinfo import ZoneInfo
    CAMPUS_TIMEZONE = ZoneInfo("Asia/Kolkata")
except Exception:
    CAMPUS_TIMEZONE = datetime.timezone(datetime.timedelta(hours=5, minutes=30))

def format_ist_time_dict(now: datetime.datetime) -> dict:
    """
    Constructs a standardized dictionary from a datetime object in Asia/Kolkata timezone.
    Extracts current hour (float and int), day of week, and weekend indicator.
    Handles midnight boundaries correctly.
    """
    day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    dow_num = now.weekday()
    dow_name = day_names[dow_num]
    is_weekend = 1 if dow_num in [5, 6] else 0
    hour_float = round(now.hour + now.minute / 60.0 + now.second / 3600.0, 4)

    return {
        "datetime": now,
        "iso_string": now.isoformat(),
        "time_str": now.strftime("%I:%M %p IST"),
        "date_str": now.strftime("%Y-%m-%d"),
        "hour_of_day": now.hour,
        "hour_float": hour_float,
        "day_of_week": dow_name,
        "day_of_week_num": dow_num,
        "is_weekend": is_weekend
    }

def get_campus_now() -> datetime.datetime:
    """Returns the current datetime localized to BIT Mesra Campus (Asia/Kolkata / IST) timezone."""
    try:
        return datetime.datetime.now(CAMPUS_TIMEZONE)
    except Exception:
        fallback_tz = datetime.timezone(datetime.timedelta(hours=5, minutes=30))
        return datetime.datetime.now(fallback_tz)

def get_current_ist_time() -> dict:
    """
    Obtains the current system time in Asia/Kolkata timezone (UTC+05:30).
    Returns a standardized dictionary containing formatted time strings,
    hour float, day of week, and weekend indicator.
    Handles timezone and initialization errors safely without duplicate implementations.
    """
    now = get_campus_now()
    return format_ist_time_dict(now)

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

def get_risk_level(probability: float) -> str:
    """Classifies risk probability into LOW, MEDIUM, or HIGH."""
    if probability <= RISK_THRESHOLDS["LOW_MAX"]:
        return "LOW"
    elif probability <= RISK_THRESHOLDS["MEDIUM_MAX"]:
        return "MEDIUM"
    else:
        return "HIGH"

# Stacking Ensemble Configuration
STACKING_META_FEATURES = [
    "physiological_stress_score",
    "location_risk_score",
    "environment_risk_score"
]
DEFAULT_BASELINE_WEIGHTS = {
    "physiological": 0.40,
    "location": 0.25,
    "environment": 0.35
}
DEFAULT_DECISION_THRESHOLD = 0.50

# BIT Mesra Campus GPS Configuration & Centralized Landmarks Database
from src.location.landmarks import (
    BIT_MESRA_CENTER,
    BIT_MESRA_LANDMARKS_DB,
    get_all_landmarks_list,
    get_landmark_by_id
)
import src.location.landmarks as landmarks

BIT_MESRA_LANDMARKS = get_all_landmarks_list()
