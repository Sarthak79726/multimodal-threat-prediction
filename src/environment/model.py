"""
Environmental Context Model Module.
Provides calibrated environmental risk scoring based on ambient situational factors
(Campus Local Time, Illumination / Lighting, and Surrounding Crowd Density).
Clearly documents that environmental features represent contextual observations.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional

# Ensure project root is in sys.path
_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import config

class EnvironmentalContextEngine:
    """
    Standardized inference engine for the Environmental Branch.
    Evaluates ambient situational vulnerability factors:
    1. Time of Day (IST: UTC+5:30)
    2. Illumination (Light vs Dark)
    3. Crowd Density (Low / Medium / High)
    Explicitly labeled as is_trained_ml=False (Rule-based context evaluator).
    """
    def __init__(self):
        self.model_name = "Contextual Rule-Based Evaluator (Time, Lighting, Crowd Density)"
        self.is_trained_ml = False
        self.target_semantics = "Ambient Situational Vulnerability Score [0.0 = Well-lit / Populated Public Space, 1.0 = Isolated Dark Night]"

    def predict_score(
        self,
        time_of_day_hour: float,
        lighting_condition: str = "Light",
        crowd_density: str = "Medium"
    ) -> float:
        """
        Computes calibrated continuous environment_risk_score in [0.05, 1.0].
        Strictly differentiates between Low, Medium, and High crowd densities across all times of day.
        """
        hour = float(time_of_day_hour) % 24.0

        # 1. Time-of-day factor (graduated campus schedule)
        if hour >= 23.0 or hour <= 4.5:
            time_factor = 0.35  # Deep late night (curfew hours)
        elif (20.0 <= hour < 23.0) or (4.5 < hour <= 6.0):
            time_factor = 0.22  # Twilight / evening transition
        else:
            time_factor = 0.05  # Standard active daytime

        # 2. Lighting factor
        is_dark = str(lighting_condition).strip().lower() == "dark"
        lighting_factor = 0.35 if is_dark else 0.08

        # 3. Crowd density factor (clear graduation)
        crowd_str = str(crowd_density).strip().lower()
        if crowd_str == "low":
            crowd_factor = 0.32  # High vulnerability when isolated
        elif crowd_str == "medium":
            crowd_factor = 0.15  # Normal presence of bystanders / friends
        else:
            crowd_factor = 0.02  # Well-populated public area

        # Calibrated baseline risk score
        calibrated_score = round(min(1.0, max(0.05, time_factor + lighting_factor + crowd_factor)), 4)
        return float(calibrated_score)

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "branch": "Environmental",
            "model_name": self.model_name,
            "is_trained_ml": self.is_trained_ml,
            "target_semantics": self.target_semantics,
            "input_sources": {
                "time_of_day": "BIT Mesra Campus Local Time (IST: UTC+5:30)",
                "lighting": "Client / User Contextual Lighting State (Light vs Dark)",
                "crowd_density": "Surrounding Density Assessment (Low / Medium / High)"
            }
        }

