"""
Unified Multimodal Threat Predictor.
Integrates Level 0 Base Models and Level 1 Stacking Meta-Learner with fallback to the Weighted Baseline.
Guarantees transparent provenance tracking:
- Model 1 (Physiological): Trained ML on WESAD clinical dataset.
- Model 2 (Spatial): Deterministic Geospatial Campus Analyzer (BIT Mesra).
- Model 3 (Environmental): Contextual Rule-Based Evaluator.
- Model 4 (Meta-Learner): Level 1 Stacking Classifier (when legitimately trained).
- Baseline: Weighted Late Fusion Reference Benchmark.
"""

import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional

# Ensure project root is in sys.path
_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import config
from src.physiological.model import PhysiologicalStressModel
from src.location.model import SpatialRiskEngine
from src.environment.model import EnvironmentalContextEngine
from src.fusion.meta_learner import WeightedLateFusionBaseline
from src.fusion.stacking_pipeline import StackingEnsemblePipeline, IneligibleProductionArtifactError

class MultimodalThreatPredictor:
    """
    Unified Inference Engine for Multimodal Threat Prediction System.
    Evaluates 3 input branches + Stacking Meta-Learner + Late Fusion Baseline.
    Guarantees strict production provenance gating:
    Rejects synthetic demo models from production inference and transparently
    reports untrained meta-learner status until valid multimodal threat data is collected.
    """
    def __init__(self, models_dir: Path = config.MODELS_DIR):
        self.models_dir = Path(models_dir)

        # 1. Base Model Wrappers
        self.physio_engine = PhysiologicalStressModel(self.models_dir / "physiological" / "physiological_model.pkl")
        self.spatial_engine = SpatialRiskEngine(self.models_dir / "location" / "location_model.pkl", use_ml_model=False)
        self.env_engine = EnvironmentalContextEngine()

        # 2. Level 1 Stacking Ensemble Pipeline (Strict Production Loading)
        stacking_artifact_path = self.models_dir / "fusion" / "stacking_ensemble.pkl"
        try:
            self.stacking_pipeline = StackingEnsemblePipeline.load_artifact(
                stacking_artifact_path,
                require_production_eligible=True
            )
        except IneligibleProductionArtifactError as e:
            print(f"[MultimodalThreatPredictor] Ineligible artifact rejected: {e}", flush=True)
            self.stacking_pipeline = None

        if self.stacking_pipeline is None:
            # Initialize pipeline with default architecture (unfitted)
            self.stacking_pipeline = StackingEnsemblePipeline()

        # 3. Explicit Reference Baseline
        self.baseline = WeightedLateFusionBaseline(threshold=config.DEFAULT_DECISION_THRESHOLD)

    def predict_physiological(self, input_dict: Dict[str, float]) -> float:
        """Computes physiological stress score P(Stress=1 | biometrics) from feature dictionary."""
        return self.physio_engine.predict_proba(input_dict)

    def predict_location(
        self,
        lat: Optional[float] = None,
        lon: Optional[float] = None,
        accuracy: Optional[float] = None,
        hour: float = 14.0,
        distance_to_security_m: Optional[float] = None,
        crime_rate: float = 0.10,
        lighting: float = 8.0,
        isolation: float = 0.10,
        loc_type: str = "administrative_hub"
    ) -> float:
        """Computes spatial risk score from GPS coordinates or landmark attributes."""
        return self.spatial_engine.predict_score(
            lat=lat,
            lon=lon,
            accuracy=accuracy,
            time_of_day_hour=hour,
            distance_to_security_m=distance_to_security_m,
            crime_rate=crime_rate,
            lighting=lighting,
            isolation=isolation,
            loc_type=loc_type
        )

    def predict_environment(
        self,
        time_of_day_hour: float,
        lighting_condition: str = "Light",
        crowd_density: str = "Medium"
    ) -> float:
        """Computes calibrated continuous environmental risk score from situational factors."""
        return self.env_engine.predict_score(
            time_of_day_hour=time_of_day_hour,
            lighting_condition=lighting_condition,
            crowd_density=crowd_density
        )

    def predict_environment_context(
        self,
        location: Optional[str] = None,
        hour: Optional[float] = None,
        day_of_week: Optional[str] = None,
        is_weekend: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Executes automatic AI environmental prediction for crowd and lighting conditions.
        Returns full prediction payload including categories, model names, confidence,
        and continuous calibrated scores.
        """
        res = self.env_engine.predict_environment(
            location=location,
            hour_of_day=hour,
            day_of_week=day_of_week,
            weekend_indicator=is_weekend
        )
        self._last_crowd_score = res.get("environmental_crowd_context_score", 0.15)
        self._last_lighting_score = res.get("environmental_lighting_context_score", 0.18)
        return res

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
        Combines 3 input modality risk scores.
        If a valid trained Stacking Meta-Learner exists, computes learned meta-model probability.
        Simultaneously computes the Weighted Late Fusion Baseline for explicit benchmark comparison.
        """
        # Normalize inputs to [0.0, 1.0]
        phys_norm = float(np.clip(physiological_stress_score, 0.0, 1.0))
        loc_norm = float(np.clip(location_risk_score, 0.0, 1.0))
        env_norm = float(np.clip(environment_risk_score, 0.0, 1.0))

        # 1. Baseline Weighted late fusion
        w_p = float(weight_physio)
        w_l = float(weight_location)
        w_e = float(weight_environment)
        total_w = w_p + w_l + w_e
        if total_w > 0:
            w_p_norm, w_l_norm, w_e_norm = w_p / total_w, w_l / total_w, w_e / total_w
        else:
            w_p_norm, w_l_norm, w_e_norm = 0.40, 0.25, 0.35

        contrib_p = w_p_norm * phys_norm
        contrib_l = w_l_norm * loc_norm
        contrib_e = w_e_norm * env_norm

        baseline_score = contrib_p + contrib_l + contrib_e
        baseline_pct = baseline_score * 100.0
        baseline_status = "RISKY" if baseline_score >= threshold else "SAFE"

        # 2. Level 1 Stacking Meta-Learner (strictly gated by production eligibility)
        is_prod_stacker_trained = (
            self.stacking_pipeline.is_trained
            and getattr(self.stacking_pipeline, "production_eligible", False)
        )

        if is_prod_stacker_trained:
            stacking_res = self.stacking_pipeline.predict(phys_norm, loc_norm, env_norm)
            active_score = stacking_res["active_threat_probability"]
            active_score_pct = stacking_res["active_threat_score_pct"]
            active_status = stacking_res["active_threat_status"]
            active_model = stacking_res["model_used"]
            meta_model_status = "TRAINED (Production-Eligible)"
            explanation = self._generate_stacking_explanation(
                active_score_pct, active_status, phys_norm, loc_norm, env_norm, stacking_res.get("stacking_explainability")
            )
            stacking_meta_prob = stacking_res.get("stacking_meta_probability")
            stacking_meta_stat = stacking_res.get("stacking_meta_status")
            stacking_explain = stacking_res.get("stacking_explainability")
        else:
            active_score = baseline_score
            active_score_pct = baseline_pct
            active_status = "ELEVATED CONTEXT RISK" if baseline_score >= threshold else "STANDARD CONTEXT"
            active_model = "Experimental Weighted Context Index"
            meta_model_status = "UNTRAINED — VALID MULTIMODAL THREAT DATASET REQUIRED"
            explanation = self._generate_baseline_explanation(
                baseline_pct, active_status, phys_norm, loc_norm, env_norm
            )
            stacking_meta_prob = None
            stacking_meta_stat = None
            stacking_explain = None

        return {
            # Base model scores
            "physiological_stress_score": phys_norm,
            "location_risk_score": loc_norm,
            "environment_risk_score": env_norm,
            # Explicit environmental branch outputs
            "environmental_crowd_context_score": getattr(self, "_last_crowd_score", 0.15),
            "environmental_lighting_context_score": getattr(self, "_last_lighting_score", 0.18),
            "environmental_model_provenance": "SYNTHETIC-DATA TRAINED MODEL (EXPERIMENTAL PREDICTION)",
            # Provenance flags
            "modalities_metadata": {
                "physiological": self.physio_engine.get_metadata(),
                "spatial": self.spatial_engine.get_metadata(),
                "environmental": self.env_engine.get_metadata()
            },
            # Active decision (either stacking or baseline)
            "threat_probability": active_score,
            "threat_score_pct": active_score_pct,
            "threat_status": active_status,
            "context_status": active_status,
            "active_model_name": active_model,
            "is_meta_learner_trained": is_prod_stacker_trained,
            "meta_model_status": meta_model_status,
            "experimental_context_index": active_score,
            "experimental_context_index_pct": active_score_pct,
            "notice": "A validated multimodal threat-prediction model is not currently available because synchronized observations and independently verified outcome labels have not yet been collected.",
            "clarification": "This is a heuristic index, not a calibrated probability of physical threat.",
            # Stacking Meta-Learner details
            "stacking_meta_probability": stacking_meta_prob,
            "stacking_meta_status": stacking_meta_stat,
            "stacking_explainability": stacking_explain,
            # Baseline details
            "baseline_threat_probability": baseline_score,
            "baseline_threat_score_pct": baseline_pct,
            "baseline_status": baseline_status,
            "baseline_weights": {
                "physiological": w_p_norm,
                "location": w_l_norm,
                "environment": w_e_norm
            },
            "contributions": {
                "physiological": contrib_p,
                "location": contrib_l,
                "environment": contrib_e
            },
            "contributions_pct": {
                "physiological": contrib_p * 100.0,
                "location": contrib_l * 100.0,
                "environment": contrib_e * 100.0
            },
            "threshold_used": threshold,
            "explanation": explanation
        }

    def _generate_stacking_explanation(
        self,
        final_pct: float,
        status: str,
        phys_norm: float,
        loc_norm: float,
        env_norm: float,
        meta_explain: Optional[Dict[str, Any]]
    ) -> str:
        coefs = meta_explain.get("coefficients", {}) if meta_explain else {}
        b0 = meta_explain.get("intercept_beta0", 0.0) if meta_explain else 0.0

        p_str = f"Physiological stress is {phys_norm*100:.1f}%."
        l_str = f"Location risk is {loc_norm*100:.1f}%."
        e_str = f"Environmental risk is {env_norm*100:.1f}%."

        coef_str = ", ".join([f"{k}: beta={v:+.3f}" for k, v in coefs.items()]) if coefs else "learned parameters"
        return (
            f"The Stacking Meta-Learner (Logistic Regression) computed an overall threat probability of {final_pct:.2f}% ({status}). "
            f"Model weights learned via Out-of-Fold training ({coef_str}, intercept={b0:+.3f}). "
            f"Input conditions: {p_str} {l_str} {e_str}"
        )

    def _generate_baseline_explanation(
        self,
        final_pct: float,
        status: str,
        phys_norm: float,
        loc_norm: float,
        env_norm: float
    ) -> str:
        return (
            f"A validated multimodal threat-prediction model is not currently available because "
            f"synchronized observations and independently verified outcome labels have not yet been collected. "
            f"Displaying Experimental Weighted Context Index ({final_pct:.2f}% / {status}). "
            f"This is a heuristic index, not a calibrated probability of physical threat."
        )

if __name__ == "__main__":
    predictor = MultimodalThreatPredictor()
    print("MultimodalThreatPredictor initialized successfully.")
    res = predictor.predict_fusion(0.20, 0.85, 0.40)
    print("Test inference result:")
    print("  Active Model:", res["active_model_name"])
    print("  Threat Score:", f"{res['threat_score_pct']:.2f}% ({res['threat_status']})")
    print("  Explanation:", res["explanation"])
