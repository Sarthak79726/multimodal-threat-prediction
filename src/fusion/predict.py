import os
import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional

# Ensure project root is in sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
import config

class MultimodalThreatPredictor:
    """
    Unified Inference Engine for Multimodal Threat Prediction System.
    Evaluates strictly 3 input branches + 1 Meta-Learner Fusion Model:
    1. Model 1: Physiological Stress Model -> physiological_stress_score
    2. Model 2: Location Risk Model -> location_risk_score
    3. Model 3: Environment Risk Model -> environment_risk_score
    4. Fusion Model -> threat_probability (SAFE or RISKY)
    """
    def __init__(self, models_dir: Path = config.MODELS_DIR):
        self.models_dir = Path(models_dir)
        self.physio_artifact = self._load_pickle(self.models_dir / "physiological" / "physiological_model.pkl")
        self.loc_artifact = self._load_pickle(self.models_dir / "location" / "location_model.pkl")
        self.env_artifact = self._load_pickle(self.models_dir / "environment" / "environment_model.pkl")
        self.fusion_artifact = self._load_pickle(self.models_dir / "fusion" / "fusion_model.pkl")
        
    def _load_pickle(self, path: Path) -> Optional[Dict[str, Any]]:
        if not path.exists():
            print(f"Warning: Model artifact at {path} does not exist yet.", flush=True)
            return None
        with open(path, "rb") as f:
            return pickle.load(f)
            
    def predict_physiological(self, input_dict: Dict[str, float]) -> float:
        """Computes physiological_stress_score from feature dictionary or model fallback."""
        if self.physio_artifact is None:
            return 0.20
            
        model = self.physio_artifact["model"]
        scaler = self.physio_artifact.get("scaler")
        feature_names = self.physio_artifact["feature_names"]
        
        row = {fname: input_dict.get(fname, 0.0) for fname in feature_names}
        X_df = pd.DataFrame([row])
        
        if scaler is not None:
            X_vec = scaler.transform(X_df)
            X_df = pd.DataFrame(X_vec, columns=feature_names)
            
        prob = float(model.predict_proba(X_df)[0, 1])
        return prob
        
    def predict_location(
        self,
        lat: Optional[float] = None,
        lon: Optional[float] = None,
        distance_to_security_m: Optional[float] = None,
        crime_rate: float = 0.1,
        lighting: float = 8.0,
        isolation: float = 0.1,
        hour: float = 14.0,
        loc_type: str = "administrative_hub"
    ) -> float:
        """Computes location_risk_score from location attributes or GPS coordinates."""
        if lat is not None and lon is not None:
            from src.location.campus_map import CampusMapAnalyzer
            analyzer = CampusMapAnalyzer()
            res = analyzer.analyze_gps(lat, lon, time_of_day_hour=hour)
            crime_rate = res.get("crime_rate_index", 0.1)
            lighting = res.get("lighting_level", 8.0)
            isolation = res.get("isolation_index", 0.1)
            loc_type = res.get("location_type", "administrative_hub")
            distance_to_security_m = res.get("distance_to_security_m", 20.0)
            if self.loc_artifact is None:
                return float(res.get("location_risk_score", 0.15))
        else:
            if self.loc_artifact is None:
                return 0.15
            lat = lat if lat is not None else 23.4123
            lon = lon if lon is not None else 85.4399
            distance_to_security_m = distance_to_security_m if distance_to_security_m is not None else 20.0

        model = self.loc_artifact["model"]
        preprocessor = self.loc_artifact["preprocessor"]

        df_in = pd.DataFrame([{
            "latitude": lat,
            "longitude": lon,
            "distance_to_security_m": distance_to_security_m,
            "crime_rate_index": crime_rate,
            "lighting_level": lighting,
            "isolation_index": isolation,
            "time_of_day_hour": hour,
            "location_type": loc_type
        }])

        X_proc = preprocessor.transform(df_in)
        prob = float(model.predict_proba(X_proc)[0, 1])
        return prob

    def predict_environment(
        self,
        time_of_day_hour: float,
        lighting_condition: str = "Light",
        crowd_density: str = "Medium"
    ) -> float:
        """
        Computes calibrated continuous environment_risk_score from environment attributes.
        Strictly differentiates between Low, Medium, and High crowd densities across all times of day.
        """
        hour = float(time_of_day_hour) % 24.0
        
        # 1. Time-of-day factor (graduated night factor)
        if hour >= 23.0 or hour <= 4.5:
            time_factor = 0.35  # Deep late night
        elif (20.0 <= hour < 23.0) or (4.5 < hour <= 6.0):
            time_factor = 0.22  # Evening / twilight
        else:
            time_factor = 0.05  # Normal daytime

        # 2. Lighting factor
        is_dark = str(lighting_condition).strip().lower() == "dark"
        lighting_factor = 0.35 if is_dark else 0.08

        # 3. Crowd density factor (clear, distinct graduation)
        crowd_str = str(crowd_density).strip().lower()
        if crowd_str == "low":
            crowd_factor = 0.32  # High vulnerability when isolated
        elif crowd_str == "medium":
            crowd_factor = 0.15  # Normal presence of bystanders / friends
        else:
            crowd_factor = 0.02  # Well-populated public area

        # Calibrated baseline risk score
        calibrated_score = round(min(1.0, max(0.05, time_factor + lighting_factor + crowd_factor)), 4)
        return calibrated_score

    def _generate_explanation(
        self,
        physio_pct: float,
        loc_pct: float,
        env_pct: float,
        physio_contrib: float,
        loc_contrib: float,
        env_contrib: float,
        total_pct: float,
        status: str
    ) -> str:
        if status == "RISKY":
            reasons = []
            if physio_pct > 50:
                reasons.append(f"high physiological stress ({physio_pct:.1f}%)")
            if loc_pct > 50:
                reasons.append(f"elevated location risk ({loc_pct:.1f}%)")
            if env_pct > 50:
                reasons.append(f"adverse environmental factors ({env_pct:.1f}%)")
            reason_str = ", ".join(reasons) if reasons else "combined multimodal risk factors"
            return (
                f"The prototype threat-risk score is elevated at {total_pct:.2f}% ({status}) driven by {reason_str}. "
                f"All three input branches contribute proportionally based on configured prototype weights."
            )
        else:
            if loc_pct > 65 and physio_pct < 30:
                return (
                    f"The prototype threat-risk score is {total_pct:.2f}% ({status} / Low-Medium Risk). Although location risk is "
                    f"elevated at {loc_pct:.1f}% (which may warrant spatial investigation), low physiological stress ({physio_pct:.1f}%) "
                    f"and moderate environmental conditions ({env_pct:.1f}%) keep the overall weighted score well below the 50.0% threshold. "
                    f"High location risk alone does not force a high threat classification without physiological or environmental evidence of danger."
                )
            else:
                return (
                    f"The prototype threat-risk score is {total_pct:.2f}% ({status}). Physiological stress ({physio_pct:.1f}%), "
                    f"location risk ({loc_pct:.1f}%), and environmental risk ({env_pct:.1f}%) combine within normal safe parameters."
                )

    def predict_fusion(
        self,
        physiological_stress_score: float,
        location_risk_score: float,
        environment_risk_score: float,
        threshold: float = 0.50,
        weight_physio: float = 0.40,
        weight_location: float = 0.25,
        weight_environment: float = 0.35
    ) -> Dict[str, Any]:
        """
        Combines 3 input risk scores into final prototype threat-risk score using an explainable
        weighted baseline model:
        fusion_score = 0.40 * physiological_risk + 0.25 * location_risk + 0.35 * environment_risk
        
        All inputs are normalized to the 0.0 - 1.0 scale before calculation.
        """
        # 1. Normalize inputs to 0.0 - 1.0 decimal range
        physio_norm = min(1.0, max(0.0, float(physiological_stress_score)))
        loc_norm = min(1.0, max(0.0, float(location_risk_score)))
        env_norm = min(1.0, max(0.0, float(environment_risk_score)))
        
        # 2. Configurable Prototype Weights (Default 40% Physio, 25% Location, 35% Environment)
        w_physio = float(weight_physio)
        w_loc = float(weight_location)
        w_env = float(weight_environment)
        
        total_w = w_physio + w_loc + w_env
        if total_w > 0:
            w_physio_norm = w_physio / total_w
            w_loc_norm = w_loc / total_w
            w_env_norm = w_env / total_w
        else:
            w_physio_norm, w_loc_norm, w_env_norm = 0.40, 0.25, 0.35
            
        # 3. Calculate individual branch contributions
        physio_contrib = w_physio_norm * physio_norm
        loc_contrib = w_loc_norm * loc_norm
        env_contrib = w_env_norm * env_norm
        
        # 4. Final Prototype Weighted Score
        fusion_score = physio_contrib + loc_contrib + env_contrib
        fusion_score_pct = fusion_score * 100.0
        
        status = "RISKY" if fusion_score >= threshold else "SAFE"
        
        # 5. Requirement #10: Detailed Debug Logging
        print(f"[DEBUG FUSION] Raw Input Scores: Physio={physiological_stress_score:.4f}, Location={location_risk_score:.4f}, Environment={environment_risk_score:.4f}", flush=True)
        print(f"[DEBUG FUSION] Normalized Inputs: Physio={physio_norm:.4f} ({physio_norm*100:.1f}%), Location={loc_norm:.4f} ({loc_norm*100:.1f}%), Environment={env_norm:.4f} ({env_norm*100:.1f}%)", flush=True)
        print(f"[DEBUG FUSION] Configured Weights: Physio={w_physio_norm:.2f}, Location={w_loc_norm:.2f}, Environment={w_env_norm:.2f}", flush=True)
        print(f"[DEBUG FUSION] Branch Contributions: Physio={physio_contrib:.4f} ({physio_contrib*100:.2f}%), Location={loc_contrib:.4f} ({loc_contrib*100:.2f}%), Environment={env_contrib:.4f} ({env_contrib*100:.2f}%)", flush=True)
        print(f"[DEBUG FUSION] Final Prototype Risk Score: {fusion_score:.4f} ({fusion_score_pct:.2f}%) | Status: {status} (Threshold: {threshold:.2f})", flush=True)
        
        # 6. Dynamic Explanation Narrative
        explanation = self._generate_explanation(
            physio_pct=physio_norm * 100.0,
            loc_pct=loc_norm * 100.0,
            env_pct=env_norm * 100.0,
            physio_contrib=physio_contrib * 100.0,
            loc_contrib=loc_contrib * 100.0,
            env_contrib=env_contrib * 100.0,
            total_pct=fusion_score_pct,
            status=status
        )
        
        return {
            "physiological_stress_score": physio_norm,
            "location_risk_score": loc_norm,
            "environment_risk_score": env_norm,
            "weights": {
                "physiological": w_physio_norm,
                "location": w_loc_norm,
                "environment": w_env_norm
            },
            "contributions": {
                "physiological": physio_contrib,
                "location": loc_contrib,
                "environment": env_contrib
            },
            "contributions_pct": {
                "physiological": physio_contrib * 100.0,
                "location": loc_contrib * 100.0,
                "environment": env_contrib * 100.0
            },
            "threat_probability": fusion_score,
            "threat_score_pct": fusion_score_pct,
            "threat_status": status,
            "threshold_used": threshold,
            "explanation": explanation
        }

if __name__ == "__main__":
    predictor = MultimodalThreatPredictor()
    
    print("--- Test 1: Main Administrative Building (Daytime) ---")
    loc_score_1 = predictor.predict_location(lat=23.4123, lon=85.4399, hour=14.0)
    fusion_res_1 = predictor.predict_fusion(0.15, loc_score_1, 0.10)
    
    print("\n--- Test 2: User Scenario (Physio 1%, Location 85%, Environment 40%) ---")
    fusion_res_test = predictor.predict_fusion(0.01, 0.85, 0.40)
    print("Resulting Score:", fusion_res_test['threat_score_pct'])


