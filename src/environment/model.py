"""
Environmental Context AI Model & Automatic Prediction Engine.
Provides automatic time and GPS-based location detection, ML-based experimental predictions
for crowd density and lighting condition, continuous lux estimation, and transparent
data provenance reporting.

Targets:
1. Model A: simulated_crowd_density_category ('Low', 'Medium', 'High')
2. Model B: simulated_lighting_category ('Dark', 'Low', 'Good')
3. Optional: simulated_lighting_lux_estimate (Continuous)

All predictions are explicitly tagged:
- SYNTHETIC-DATA TRAINED MODEL
- EXPERIMENTAL PREDICTION
- NOT LIVE SENSOR MEASUREMENT
"""

import sys
import pickle
import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo  # type: ignore

import numpy as np
import pandas as pd

_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import config
from src.location.campus_map import CampusMapAnalyzer, haversine_distance
from src.location.landmarks import get_all_landmarks_list

SUPPORTED_LOCATIONS = [
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

class EnvironmentalContextEngine:
    """
    Automatic Environmental Context Inference Engine.
    Combines:
    1. Automatic IST system time detection (Asia/Kolkata).
    2. Automatic GPS coordinate resolution against BIT Mesra landmarks (with geofencing).
    3. Machine Learning inference for Crowd Density (Model A) and Lighting Condition (Model B).
    4. Continuous lux estimation via tree-based regressor.
    5. Calibrated environmental context vulnerability scoring.
    """

    def __init__(
        self,
        crowd_model_path: Optional[Path] = None,
        lighting_model_path: Optional[Path] = None
    ):
        self.crowd_model_path = crowd_model_path or (config.MODELS_DIR / "environment" / "crowd_model.pkl")
        self.lighting_model_path = lighting_model_path or (config.MODELS_DIR / "environment" / "lighting_model.pkl")
        
        self.crowd_artifact: Optional[Dict[str, Any]] = None
        self.lighting_artifact: Optional[Dict[str, Any]] = None
        self.is_trained_ml: bool = False
        self.load_error: Optional[str] = None

        self.campus_analyzer = CampusMapAnalyzer()
        self.landmarks = get_all_landmarks_list()

        self._load_models()

    def _load_models(self) -> None:
        """Loads trained ML artifacts for crowd and lighting prediction."""
        try:
            if self.crowd_model_path.exists() and self.lighting_model_path.exists():
                with open(self.crowd_model_path, "rb") as f:
                    self.crowd_artifact = pickle.load(f)
                with open(self.lighting_model_path, "rb") as f:
                    self.lighting_artifact = pickle.load(f)

                self.is_trained_ml = True
                self.load_error = None
            else:
                self.is_trained_ml = False
                self.load_error = f"Model artifacts not found at {self.crowd_model_path} or {self.lighting_model_path}"
        except Exception as e:
            self.is_trained_ml = False
            self.load_error = f"Failed to deserialize artifacts: {str(e)}"
            self.crowd_artifact = None
            self.lighting_artifact = None

    @staticmethod
    def get_current_ist_time() -> Dict[str, Any]:
        """
        Obtains the current system time in Asia/Kolkata timezone.
        Extracts current hour (float and int), day of week, and weekend indicator.
        Handles midnight boundaries correctly.
        """
        try:
            tz = ZoneInfo("Asia/Kolkata")
            now = datetime.datetime.now(tz)
        except Exception:
            # Fallback using config helper or timedelta
            now = config.get_campus_now()

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

    def resolve_campus_location(
        self,
        lat: Optional[float] = None,
        lon: Optional[float] = None,
        max_geofence_meters: float = 350.0
    ) -> Dict[str, Any]:
        """
        Resolves GPS coordinates against BIT Mesra campus landmarks.
        Returns resolved location if within geofence; otherwise flags as outside campus
        or GPS unavailable, enabling safe manual location selection.
        """
        if lat is None or lon is None:
            return {
                "status": "GPS_UNAVAILABLE",
                "resolved_location": None,
                "distance_meters": None,
                "is_within_geofence": False,
                "message": "GPS coordinates not available. Manual campus location selection required."
            }

        # Calculate distance to all known campus landmarks
        min_dist = float("inf")
        nearest_landmark = None

        for lm in self.landmarks:
            dist = haversine_distance(lat, lon, lm["latitude"], lm["longitude"])
            if dist < min_dist:
                min_dist = dist
                nearest_landmark = lm

        if nearest_landmark is None:
            return {
                "status": "ERROR",
                "resolved_location": None,
                "distance_meters": None,
                "is_within_geofence": False,
                "message": "No campus landmarks configured."
            }

        # Geofence threshold check: either landmark's specific radius or max threshold
        effective_threshold = max(float(nearest_landmark.get("radius", 150.0)), max_geofence_meters)
        is_within_geofence = min_dist <= effective_threshold

        # Ensure exact name match against supported locations
        matched_name = nearest_landmark["name"]
        if matched_name not in SUPPORTED_LOCATIONS:
            # Map alias if applicable
            for sup in SUPPORTED_LOCATIONS:
                if sup.lower() in matched_name.lower() or matched_name.lower() in sup.lower():
                    matched_name = sup
                    break

        if is_within_geofence:
            return {
                "status": "RESOLVED_GPS",
                "resolved_location": matched_name,
                "landmark_id": nearest_landmark["id"],
                "distance_meters": round(min_dist, 1),
                "is_within_geofence": True,
                "message": f"Successfully resolved to '{matched_name}' ({int(min_dist)}m from center)."
            }
        else:
            return {
                "status": "OUTSIDE_CAMPUS_GEOFENCE",
                "resolved_location": None,
                "nearest_landmark_candidate": matched_name,
                "distance_meters": round(min_dist, 1),
                "is_within_geofence": False,
                "message": f"Coordinates are {int(min_dist)}m from campus ({matched_name}), exceeding supported geofence ({int(effective_threshold)}m). Manual location required."
            }

    def predict_environment(
        self,
        location: Optional[str] = None,
        hour_of_day: Optional[float] = None,
        day_of_week: Optional[str] = None,
        weekend_indicator: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Executes automatic environmental predictions using trained Model A and Model B.
        If temporal parameters are omitted, automatically extracts current IST system time.
        Validates location against supported BIT Mesra landmarks.
        """
        # Automatic time detection if parameters not provided
        if hour_of_day is None or day_of_week is None or weekend_indicator is None:
            ist_info = self.get_current_ist_time()
            hour_of_day = ist_info["hour_of_day"] if hour_of_day is None else hour_of_day
            day_of_week = ist_info["day_of_week"] if day_of_week is None else day_of_week
            weekend_indicator = ist_info["is_weekend"] if weekend_indicator is None else weekend_indicator
            time_display = ist_info["time_str"]
        else:
            int_h = int(hour_of_day) % 24
            min_frac = int((hour_of_day - int_h) * 60)
            ampm = "AM" if int_h < 12 else "PM"
            disp_h = 12 if (int_h % 12 == 0) else (int_h % 12)
            time_display = f"{disp_h:02d}:{min_frac:02d} {ampm} IST"

        # Validate location
        clean_loc = None
        if location is not None and str(location).strip():
            target_str = str(location).strip()
            for sup in SUPPORTED_LOCATIONS:
                if sup.lower() == target_str.lower():
                    clean_loc = sup
                    break
            # Partial match check
            if clean_loc is None:
                for sup in SUPPORTED_LOCATIONS:
                    if sup.lower() in target_str.lower() or target_str.lower() in sup.lower():
                        clean_loc = sup
                        break

        if clean_loc is None:
            return {
                "status": "UNKNOWN",
                "error": "UNSUPPORTED_LOCATION",
                "message": f"Location '{location}' is not a recognized BIT Mesra campus landmark. Must be one of 17 supported locations.",
                "location": "UNKNOWN",
                "time_display": time_display,
                "hour_of_day": hour_of_day,
                "day_of_week": day_of_week,
                "weekend_indicator": weekend_indicator,
                "crowd_prediction": {
                    "category": "UNKNOWN",
                    "confidence": 0.0,
                    "model_name": "None",
                    "status": "UNAVAILABLE"
                },
                "lighting_prediction": {
                    "category": "UNKNOWN",
                    "confidence": 0.0,
                    "model_name": "None",
                    "lux_estimate": None,
                    "status": "UNAVAILABLE"
                },
                "environmental_crowd_context_score": 0.15,
                "environmental_lighting_context_score": 0.18,
                "calibrated_environmental_score": 0.20,
                "is_trained_ml": False,
                "synthetic_provenance": "NONE",
                "warning": "Location unknown. Output cannot be calculated from ML models."
            }

        # Check if models are loaded
        if not self.is_trained_ml or self.crowd_artifact is None or self.lighting_artifact is None:
            # Fall back safely to rule-based calculation
            fallback_score = self.predict_score(hour_of_day, "Good", "Medium")
            return {
                "status": "UNKNOWN",
                "error": "MODEL_UNAVAILABLE",
                "message": f"ML model artifacts not loaded ({self.load_error}).",
                "location": clean_loc,
                "time_display": time_display,
                "hour_of_day": hour_of_day,
                "day_of_week": day_of_week,
                "weekend_indicator": weekend_indicator,
                "crowd_prediction": {
                    "category": "UNKNOWN",
                    "confidence": 0.0,
                    "model_name": "Fallback Evaluator",
                    "status": "MODEL_MISSING"
                },
                "lighting_prediction": {
                    "category": "UNKNOWN",
                    "confidence": 0.0,
                    "model_name": "Fallback Evaluator",
                    "lux_estimate": None,
                    "status": "MODEL_MISSING"
                },
                "environmental_crowd_context_score": 0.15,
                "environmental_lighting_context_score": 0.18,
                "calibrated_environmental_score": fallback_score,
                "is_trained_ml": False,
                "synthetic_provenance": "NONE",
                "warning": "Model artifacts unavailable. Displaying UNKNOWN status."
            }

        # Prepare feature DataFrame for preprocessing
        input_df = pd.DataFrame([{
            "location": clean_loc,
            "hour_of_day": float(hour_of_day),
            "day_of_week": str(day_of_week),
            "weekend_indicator": int(weekend_indicator)
        }])

        try:
            # Preprocess features using the trained preprocessor
            preprocessor = self.crowd_artifact["preprocessor"]
            X_features = preprocessor.transform(input_df)

            # 1. Model A: Crowd Density Prediction
            crowd_model = self.crowd_artifact["estimator"]
            crowd_classes = self.crowd_artifact["classes"]
            crowd_probs = crowd_model.predict_proba(X_features)[0]
            crowd_pred_idx = int(np.argmax(crowd_probs))
            crowd_pred_cat = crowd_classes[crowd_pred_idx]
            crowd_conf = float(crowd_probs[crowd_pred_idx])
            crowd_model_name = self.crowd_artifact["model_name"]

            # 2. Model B: Lighting Condition Prediction
            light_model = self.lighting_artifact["estimator"]
            light_classes = self.lighting_artifact["classes"]
            light_probs = light_model.predict_proba(X_features)[0]
            light_pred_idx = int(np.argmax(light_probs))
            light_pred_cat = light_classes[light_pred_idx]
            light_conf = float(light_probs[light_pred_idx])
            light_model_name = self.lighting_artifact["model_name"]

            # 3. Continuous Lux Estimation (Regression)
            lux_regressor = self.lighting_artifact.get("lux_regressor")
            if lux_regressor is not None:
                lux_val = round(float(lux_regressor.predict(X_features)[0]), 1)
            else:
                # Approximate default based on predicted class
                lux_val = 4.0 if light_pred_cat == "Dark" else (55.0 if light_pred_cat == "Low" else 500.0)

            # 4. Calibrated Continuous Context Vulnerability Scores
            # Crowd Vulnerability: Low isolation = high vulnerability (0.32), High crowd = low vulnerability (0.02)
            if crowd_pred_cat == "Low":
                crowd_vuln = 0.32
            elif crowd_pred_cat == "Medium":
                crowd_vuln = 0.15
            else:
                crowd_vuln = 0.02

            # Lighting Vulnerability: Dark = 0.35, Low = 0.18, Good = 0.05
            if light_pred_cat == "Dark":
                light_vuln = 0.35
            elif light_pred_cat == "Low":
                light_vuln = 0.18
            else:
                light_vuln = 0.05

            # Diurnal Time Vulnerability Factor
            hour_mod = float(hour_of_day) % 24.0
            if hour_mod >= 23.0 or hour_mod <= 4.5:
                time_factor = 0.35
            elif (20.0 <= hour_mod < 23.0) or (4.5 < hour_mod <= 6.0):
                time_factor = 0.22
            else:
                time_factor = 0.05

            calibrated_env_score = round(min(1.0, max(0.05, time_factor + light_vuln + crowd_vuln)), 4)

            return {
                "status": "SUCCESS",
                "location": clean_loc,
                "time_display": time_display,
                "hour_of_day": round(float(hour_of_day), 2),
                "day_of_week": day_of_week,
                "weekend_indicator": weekend_indicator,
                "crowd_prediction": {
                    "category": crowd_pred_cat,
                    "confidence": round(crowd_conf, 4),
                    "confidence_pct": round(crowd_conf * 100, 1),
                    "probabilities": {cls: round(float(prob), 4) for cls, prob in zip(crowd_classes, crowd_probs)},
                    "model_name": crowd_model_name,
                    "status": "ACTIVE_ML_PREDICTION",
                    "provenance": "SYNTHETIC-DATA TRAINED MODEL (EXPERIMENTAL)"
                },
                "lighting_prediction": {
                    "category": light_pred_cat,
                    "confidence": round(light_conf, 4),
                    "confidence_pct": round(light_conf * 100, 1),
                    "probabilities": {cls: round(float(prob), 4) for cls, prob in zip(light_classes, light_probs)},
                    "model_name": light_model_name,
                    "lux_estimate": lux_val,
                    "status": "ACTIVE_ML_PREDICTION",
                    "provenance": "SYNTHETIC-DATA TRAINED MODEL (EXPERIMENTAL)"
                },
                "environmental_crowd_context_score": crowd_vuln,
                "environmental_lighting_context_score": light_vuln,
                "calibrated_environmental_score": calibrated_env_score,
                "is_trained_ml": True,
                "synthetic_provenance": "SYNTHETIC_SIMULATION_RULE_GENERATED",
                "warning": "EXPERIMENTAL PREDICTION — NOT LIVE SENSOR MEASUREMENT. Actual campus crowd and lighting conditions may differ from predictions."
            }

        except Exception as e:
            return {
                "status": "UNKNOWN",
                "error": "INFERENCE_FAILED",
                "message": f"Inference execution failed: {str(e)}",
                "location": clean_loc,
                "time_display": time_display,
                "hour_of_day": hour_of_day,
                "day_of_week": day_of_week,
                "weekend_indicator": weekend_indicator,
                "crowd_prediction": {"category": "UNKNOWN", "confidence": 0.0, "model_name": "None", "status": "ERROR"},
                "lighting_prediction": {"category": "UNKNOWN", "confidence": 0.0, "model_name": "None", "lux_estimate": None, "status": "ERROR"},
                "environmental_crowd_context_score": 0.15,
                "environmental_lighting_context_score": 0.18,
                "calibrated_environmental_score": 0.20,
                "is_trained_ml": False,
                "synthetic_provenance": "NONE",
                "warning": "Model inference failed. Output is UNKNOWN."
            }

    def predict_score(
        self,
        time_of_day_hour: float,
        lighting_condition: str = "Light",
        crowd_density: str = "Medium"
    ) -> float:
        """
        Backward-compatible method for continuous environmental vulnerability scoring.
        Accepts direct category strings or hour float and evaluates calibrated score.
        """
        hour = float(time_of_day_hour) % 24.0

        if hour >= 23.0 or hour <= 4.5:
            time_factor = 0.35
        elif (20.0 <= hour < 23.0) or (4.5 < hour <= 6.0):
            time_factor = 0.22
        else:
            time_factor = 0.05

        is_dark = str(lighting_condition).strip().lower() in ["dark", "low"]
        lighting_factor = 0.35 if str(lighting_condition).strip().lower() == "dark" else (0.18 if str(lighting_condition).strip().lower() == "low" else 0.05)

        crowd_str = str(crowd_density).strip().lower()
        if crowd_str == "low":
            crowd_factor = 0.32
        elif crowd_str == "medium":
            crowd_factor = 0.15
        else:
            crowd_factor = 0.02

        calibrated_score = round(min(1.0, max(0.05, time_factor + lighting_factor + crowd_factor)), 4)
        return float(calibrated_score)

    def get_metadata(self) -> Dict[str, Any]:
        """Returns metadata and provenance for the environmental branch."""
        return {
            "branch": "Environmental",
            "model_type": "AI-Based Experimental Predictor (Model A: Crowd, Model B: Lighting)",
            "is_trained_ml": self.is_trained_ml,
            "synthetic_provenance": "SYNTHETIC_SIMULATION_RULE_GENERATED",
            "disclaimer": "SYNTHETIC-DATA TRAINED MODEL — EXPERIMENTAL PREDICTION — NOT LIVE SENSOR MEASUREMENT",
            "crowd_model": self.crowd_artifact["model_name"] if self.crowd_artifact else "Unloaded",
            "lighting_model": self.lighting_artifact["model_name"] if self.lighting_artifact else "Unloaded",
            "supported_locations_count": len(SUPPORTED_LOCATIONS),
            "timezone": "Asia/Kolkata (IST: UTC+5:30)"
        }
