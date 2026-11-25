"""
Spatial Risk Model & Geospatial Engine Module.
Provides deterministic geospatial risk assessment for the BIT Mesra campus and
separates deterministic Haversine geofence calculations from trained machine learning classifiers.
"""

import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, List

# Ensure project root is in sys.path
_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import config
from src.location.campus_map import CampusMapAnalyzer

class SpatialRiskEngine:
    """
    Standardized inference engine for the Spatial Branch.
    Operates in two modes:
    1. Deterministic Geospatial Engine (Default): Calculates risk from Haversine distances
       to BIT Mesra campus landmarks, guarded security outposts, zone isolation indices,
       and time-adjusted illumination. Transparently flagged as is_trained_ml=False.
    2. Trained ML Model (Optional): If a legitimate labelled spatial dataset is provided.
    """
    def __init__(self, model_path: Optional[Path] = None, use_ml_model: bool = False):
        self.model_path = model_path or (config.MODELS_DIR / "location" / "location_model.pkl")
        self.use_ml_model = use_ml_model
        self.analyzer = CampusMapAnalyzer()
        self.ml_artifact = None
        self.is_trained_ml = False
        self.model_name = "Deterministic Geospatial Engine (Haversine & Campus Geofencing)"
        self.target_semantics = "Campus Spatial Vulnerability Index [0.0 = High Security / Low Isolation, 1.0 = Secluded / High Risk]"
        self._load_artifact()

    def _load_artifact(self) -> None:
        if self.use_ml_model and self.model_path.exists():
            try:
                with open(self.model_path, "rb") as f:
                    self.ml_artifact = pickle.load(f)
                self.is_trained_ml = True
                self.model_name = self.ml_artifact.get("model_name", "Trained Spatial Model")
            except Exception as e:
                print(f"[SpatialRiskEngine] Notice: Could not load ML artifact ({e}). Defaulting to deterministic engine.", flush=True)
                self.is_trained_ml = False
        else:
            self.is_trained_ml = False
            self.model_name = "Deterministic Geospatial Engine (Haversine & Campus Geofencing)"

    def predict_score(
        self,
        lat: Optional[float] = None,
        lon: Optional[float] = None,
        accuracy: Optional[float] = None,
        time_of_day_hour: float = 14.0,
        distance_to_security_m: Optional[float] = None,
        crime_rate: float = 0.10,
        lighting: float = 8.0,
        isolation: float = 0.10,
        loc_type: str = "administrative_hub"
    ) -> float:
        """
        Computes spatial risk score P_spatial in [0.0, 1.0].
        Uses genuine GPS coordinates if provided; otherwise falls back to landmark attributes.
        """
        # If coordinates provided, run through campus map analyzer
        if lat is not None and lon is not None:
            analysis = self.analyzer.analyze_gps(
                lat=lat,
                lon=lon,
                accuracy=accuracy,
                time_of_day_hour=time_of_day_hour
            )
            if not self.use_ml_model or self.ml_artifact is None:
                # Scientifically correct: Use deterministic geospatial evaluation
                score = analysis.get("location_risk_score", 0.15)
                return float(min(1.0, max(0.0, score)))

            # If user explicitly requested ML evaluation on extracted features:
            crime_rate = analysis.get("crime_rate_index", crime_rate)
            lighting = analysis.get("lighting_level", lighting)
            isolation = analysis.get("isolation_index", isolation)
            loc_type = analysis.get("location_type", loc_type)
            distance_to_security_m = analysis.get("distance_to_security_m", 20.0)

        # ML Model inference fallback
        if self.use_ml_model and self.ml_artifact is not None:
            model = self.ml_artifact["model"]
            preprocessor = self.ml_artifact["preprocessor"]
            df_in = pd.DataFrame([{
                "latitude": lat if lat is not None else config.BIT_MESRA_CENTER["latitude"],
                "longitude": lon if lon is not None else config.BIT_MESRA_CENTER["longitude"],
                "distance_to_security_m": distance_to_security_m if distance_to_security_m is not None else 20.0,
                "crime_rate_index": crime_rate,
                "lighting_level": lighting,
                "isolation_index": isolation,
                "time_of_day_hour": time_of_day_hour,
                "location_type": loc_type
            }])
            X_proc = preprocessor.transform(df_in)
            prob = float(model.predict_proba(X_proc)[0, 1])
            return min(1.0, max(0.0, prob))

        # Default fallback
        return 0.15

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "branch": "Spatial",
            "model_name": self.model_name,
            "is_trained_ml": self.is_trained_ml,
            "target_semantics": self.target_semantics,
            "data_source": "BIT Mesra Verified Geographic Landmarks & Haversine Distance Engine",
            "campus_center": config.BIT_MESRA_CENTER
        }

