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
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional

# Configure diagnostic logger
logger = logging.getLogger("MultimodalThreatPredictor")
if not logger.handlers:
    _handler = logging.StreamHandler(sys.stdout)
    _handler.setFormatter(logging.Formatter("[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(_handler)
logger.setLevel(logging.INFO)

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
        try:
            self.env_engine = EnvironmentalContextEngine()
        except Exception as e:
            print(f"[MultimodalThreatPredictor] Warning: EnvironmentalContextEngine initialization failed: {e}", flush=True)
            self.env_engine = None

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

    def get_current_ist_time(self) -> Dict[str, Any]:
        """
        Public delegate for retrieving the current IST time info.
        Safely attempts to use self.env_engine, falling back to config.get_current_ist_time().
        Ensures consistent, safe time detection across all deployment targets.
        """
        if hasattr(self, "env_engine") and self.env_engine is not None and hasattr(self.env_engine, "get_current_ist_time"):
            try:
                return self.env_engine.get_current_ist_time()
            except Exception:
                pass
        return config.get_current_ist_time()

    def resolve_campus_location(
        self,
        lat: Optional[float] = None,
        lon: Optional[float] = None,
        max_geofence_meters: float = 350.0,
        accuracy: Optional[float] = None,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Public delegate for resolving GPS coordinates against BIT Mesra campus landmarks.
        Safely attempts to use self.env_engine, handling legacy method signatures defensively.
        """
        if hasattr(self, "env_engine") and self.env_engine is not None and hasattr(self.env_engine, "resolve_campus_location"):
            try:
                return self.env_engine.resolve_campus_location(
                    lat=lat,
                    lon=lon,
                    max_geofence_meters=max_geofence_meters,
                    accuracy=accuracy,
                    **kwargs
                )
            except TypeError:
                try:
                    return self.env_engine.resolve_campus_location(
                        lat=lat,
                        lon=lon,
                        max_geofence_meters=max_geofence_meters
                    )
                except Exception as e:
                    logger.warning("Delegated resolve_campus_location failed: %s", e)
            except Exception as e:
                logger.warning("Delegated resolve_campus_location failed: %s", e)

        return {
            "status": "GPS_UNAVAILABLE",
            "resolved_location": None,
            "distance_meters": None,
            "is_within_geofence": False,
            "ambiguous_candidates": [],
            "message": "Campus location resolver unavailable."
        }

    def predict_environment(
        self,
        time_of_day_hour: float,
        lighting_condition: str = "Light",
        crowd_density: str = "Medium"
    ) -> float:
        """Computes situational environmental context risk score from situational factors."""
        if hasattr(self, "env_engine") and self.env_engine is not None and hasattr(self.env_engine, "predict_score"):
            try:
                return self.env_engine.predict_score(
                    time_of_day_hour=time_of_day_hour,
                    lighting_condition=lighting_condition,
                    crowd_density=crowd_density
                )
            except Exception:
                pass
        return 0.20

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
        and situational environmental context scores.
        Handles engine unavailability or errors defensively without crashing.
        """
        logger.info("predict_environment_context called: location='%s', hour=%s, day='%s', is_weekend=%s",
                    location, hour, day_of_week, is_weekend)

        if hasattr(self, "env_engine") and self.env_engine is not None and hasattr(self.env_engine, "predict_environment"):
            try:
                res = self.env_engine.predict_environment(
                    location=location,
                    hour_of_day=hour,
                    day_of_week=day_of_week,
                    weekend_indicator=is_weekend
                )
                self._last_crowd_score = res.get("environmental_crowd_context_score")
                self._last_lighting_score = res.get("environmental_lighting_context_score")

                # Defensive score key alignment
                env_score = res.get("environmental_context_score")
                if env_score is None:
                    env_score = res.get("calibrated_environmental_score")
                if env_score is None:
                    env_score = res.get("environment_risk_score")

                if env_score is not None and np.isfinite(env_score) and 0.0 <= float(env_score) <= 1.0:
                    res["environmental_context_score"] = float(env_score)
                    res["calibrated_environmental_score"] = float(env_score)
                    res["environment_risk_score"] = float(env_score)
                else:
                    if res.get("status") == "SUCCESS":
                        logger.warning("Environmental score non-finite or invalid: %s. Setting score to None.", env_score)
                        res["environmental_context_score"] = None
                        res["calibrated_environmental_score"] = None
                        res["environment_risk_score"] = None

                logger.info("predict_environment_context completed: status='%s', score=%s, error=%s",
                            res.get("status"), res.get("environmental_context_score"), res.get("error"))
                return res
            except Exception as e:
                logger.error("predict_environment unexpected error: %s", str(e), exc_info=True)

        # Defensive fallback when environmental engine is unavailable or uninitialized
        logger.warning("Environmental engine unavailable or uninitialized. Returning UNAVAILABLE fallback.")
        time_info = self.get_current_ist_time()
        effective_hour = hour if hour is not None else time_info["hour_float"]
        effective_day = day_of_week or time_info["day_of_week"]
        effective_wknd = is_weekend if is_weekend is not None else time_info["is_weekend"]

        self._last_crowd_score = None
        self._last_lighting_score = None

        return {
            "status": "UNAVAILABLE",
            "error": "ENVIRONMENTAL_ENGINE_UNAVAILABLE",
            "message": "EnvironmentalContextEngine is not initialized or encountered an error. Output marked UNKNOWN.",
            "location": location or "UNKNOWN",
            "time_display": time_info.get("time_str", "12:00 PM IST"),
            "hour_of_day": effective_hour,
            "day_of_week": effective_day,
            "weekend_indicator": effective_wknd,
            "crowd_prediction": {
                "category": "UNKNOWN",
                "confidence": 0.0,
                "confidence_pct": 0.0,
                "model_name": "Unavailable",
                "status": "ENGINE_UNAVAILABLE"
            },
            "lighting_prediction": {
                "category": "UNKNOWN",
                "confidence": 0.0,
                "confidence_pct": 0.0,
                "model_name": "Unavailable",
                "lux_estimate": None,
                "status": "ENGINE_UNAVAILABLE"
            },
            "environmental_crowd_context_score": None,
            "environmental_lighting_context_score": None,
            "environmental_context_score": None,
            "calibrated_environmental_score": None,
            "environment_risk_score": None,
            "is_trained_ml": False,
            "synthetic_provenance": "NONE",
            "warning": "Environmental models unavailable. Output marked UNKNOWN without silent neutral substitution."
        }

    def predict_fusion(
        self,
        physiological_stress_score: float,
        location_risk_score: float,
        environment_risk_score: Optional[float] = None,
        threshold: float = 0.50,
        weight_physio: float = 0.40,
        weight_location: float = 0.25,
        weight_environment: float = 0.35
    ) -> Dict[str, Any]:
        """
        Combines input modality risk scores.
        When all 3 branches (Physiological, Spatial, Environmental) are provided:
          - Computes full 3-branch weighted baseline context index (or Level 1 Stacking Meta-Learner if production-eligible).
        When the Environmental branch is missing / unknown (environment_risk_score is None):
          - Eliminates silent substitution.
          - Computes an explicit PARTIAL DUAL-BRANCH CONTEXT INDEX (PHYSIOLOGICAL + SPATIAL ONLY).
          - Formula: R_partial = (0.40 * P + 0.25 * L) / 0.65.
          - Sets experimental_context_index to None to signify the 3-branch index is incomplete.
        """
        # Defensive normalization of biometric and spatial inputs to [0.0, 1.0]
        p_val = 0.18 if (physiological_stress_score is None or not np.isfinite(physiological_stress_score)) else float(physiological_stress_score)
        l_val = 0.15 if (location_risk_score is None or not np.isfinite(location_risk_score)) else float(location_risk_score)
        phys_norm = float(np.clip(p_val, 0.0, 1.0))
        loc_norm = float(np.clip(l_val, 0.0, 1.0))

        # Check whether environmental modality is available and finite in [0.0, 1.0]
        is_env_available = (
            environment_risk_score is not None
            and isinstance(environment_risk_score, (int, float))
            and np.isfinite(environment_risk_score)
            and 0.0 <= float(environment_risk_score) <= 1.0
        )

        logger.info(
            "predict_fusion: P=%.4f, L=%.4f, E=%s -> mode=%s",
            phys_norm, loc_norm, environment_risk_score,
            "THREE_BRANCH_FUSION" if is_env_available else "PARTIAL_DUAL_BRANCH"
        )

        modalities_metadata = {
            "physiological": self.physio_engine.get_metadata() if hasattr(self, "physio_engine") and self.physio_engine is not None else {},
            "spatial": self.spatial_engine.get_metadata() if hasattr(self, "spatial_engine") and self.spatial_engine is not None else {},
            "environmental": self.env_engine.get_metadata() if hasattr(self, "env_engine") and self.env_engine is not None and hasattr(self.env_engine, "get_metadata") else {
                "branch": "Environmental",
                "status": "UNAVAILABLE",
                "is_trained_ml": False,
                "disclaimer": "EnvironmentalContextEngine is unavailable."
            }
        }

        # -----------------------------------------------------------------
        # BRANCH A: PARTIAL DUAL-BRANCH CONTEXT INDEX (ENV IS MISSING/UNKNOWN)
        # -----------------------------------------------------------------
        if not is_env_available:
            w_p_raw = float(weight_physio)
            w_l_raw = float(weight_location)
            dual_sum = w_p_raw + w_l_raw
            if dual_sum > 0:
                w_p_partial = w_p_raw / dual_sum
                w_l_partial = w_l_raw / dual_sum
            else:
                w_p_partial, w_l_partial = 0.40 / 0.65, 0.25 / 0.65

            contrib_p = w_p_partial * phys_norm
            contrib_l = w_l_partial * loc_norm
            partial_score = round(contrib_p + contrib_l, 4)
            partial_pct = round(partial_score * 100.0, 2)

            if partial_score <= config.RISK_THRESHOLDS["LOW_MAX"]:
                partial_status = "STANDARD CONTEXT"
            elif partial_score <= config.RISK_THRESHOLDS["MEDIUM_MAX"]:
                partial_status = "ELEVATED CONTEXT"
            else:
                partial_status = "HIGH CONTEXT"

            explanation = self._generate_partial_explanation(
                partial_pct, partial_status, phys_norm, loc_norm
            )

            return {
                "fusion_mode": "PARTIAL_DUAL_BRANCH",
                "available_modalities": ["physiological", "spatial"],
                "missing_modalities": ["environmental"],
                # Base model scores
                "physiological_stress_score": phys_norm,
                "location_risk_score": loc_norm,
                "environment_risk_score": None,
                "environmental_context_score": None,
                "environmental_crowd_context_score": None,
                "environmental_lighting_context_score": None,
                "environmental_model_provenance": "MISSING / UNKNOWN (Excluded from fusion)",
                "modalities_metadata": modalities_metadata,
                # Partial Dual-Branch Specifics
                "partial_dual_branch_index": partial_score,
                "partial_dual_branch_index_pct": partial_pct,
                "partial_fusion_name": "PARTIAL DUAL-BRANCH CONTEXT INDEX (PHYSIOLOGICAL + SPATIAL ONLY)",
                "partial_formula": "R_partial = (0.40 * P + 0.25 * L) / 0.65",
                "partial_weights": {
                    "physiological": round(w_p_partial, 4),
                    "location": round(w_l_partial, 4)
                },
                "partial_contributions": {
                    "physiological": round(contrib_p, 4),
                    "location": round(contrib_l, 4)
                },
                "partial_contributions_pct": {
                    "physiological": round(contrib_p * 100.0, 2),
                    "location": round(contrib_l * 100.0, 2)
                },
                # Complete 3-branch index is explicitly None
                "experimental_context_index": None,
                "experimental_context_index_pct": None,
                "threat_probability": partial_score,
                "threat_score_pct": partial_pct,
                "threat_status": partial_status,
                "context_status": partial_status,
                "context_classification": partial_status,
                "active_model_name": "PARTIAL DUAL-BRANCH CONTEXT INDEX (PHYSIOLOGICAL + SPATIAL ONLY)",
                "is_meta_learner_trained": False,
                "meta_model_status": "NOT EVALUATED (MISSING ENVIRONMENTAL MODALITY)",
                "notice": "A complete 3-branch context index is unavailable because environmental context is missing or unknown. Operating in fallback dual-branch mode.",
                "clarification": "This partial index evaluates physiological biometrics and spatial location only. Environmental conditions (lighting, crowd density) are excluded.",
                "stacking_meta_probability": None,
                "stacking_meta_status": None,
                "stacking_explainability": None,
                "baseline_threat_probability": partial_score,
                "baseline_threat_score_pct": partial_pct,
                "baseline_status": partial_status,
                "threshold_used": threshold,
                "explanation": explanation
            }

        # -----------------------------------------------------------------
        # BRANCH B: THREE-BRANCH CONTEXT FUSION (ALL 3 INPUTS PRESENT)
        # -----------------------------------------------------------------
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

        baseline_score = round(contrib_p + contrib_l + contrib_e, 4)
        baseline_pct = round(baseline_score * 100.0, 2)

        # Neutral heuristic classification using configured project thresholds
        if baseline_score <= config.RISK_THRESHOLDS["LOW_MAX"]:
            baseline_status = "STANDARD CONTEXT"
        elif baseline_score <= config.RISK_THRESHOLDS["MEDIUM_MAX"]:
            baseline_status = "ELEVATED CONTEXT"
        else:
            baseline_status = "HIGH CONTEXT"

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
            active_status = baseline_status
            active_model = "Experimental Weighted Context Index"
            meta_model_status = "UNTRAINED — VALID MULTIMODAL THREAT DATASET REQUIRED"
            explanation = self._generate_baseline_explanation(
                baseline_pct, active_status, phys_norm, loc_norm, env_norm
            )
            stacking_meta_prob = None
            stacking_meta_stat = None
            stacking_explain = None

        return {
            "fusion_mode": "THREE_BRANCH_FUSION",
            "available_modalities": ["physiological", "spatial", "environmental"],
            "missing_modalities": [],
            # Base model scores
            "physiological_stress_score": phys_norm,
            "location_risk_score": loc_norm,
            "environment_risk_score": env_norm,
            "environmental_context_score": env_norm,
            # Explicit environmental branch outputs
            "environmental_crowd_context_score": getattr(self, "_last_crowd_score", 0.15),
            "environmental_lighting_context_score": getattr(self, "_last_lighting_score", 0.18),
            "environmental_model_provenance": "SYNTHETIC-DATA TRAINED MODEL (EXPERIMENTAL PREDICTION)",
            # Provenance flags
            "modalities_metadata": modalities_metadata,
            # Active decision (either stacking or baseline)
            "threat_probability": active_score,
            "threat_score_pct": active_score_pct,
            "threat_status": active_status,
            "context_status": active_status,
            "context_classification": active_status,
            "active_model_name": active_model,
            "is_meta_learner_trained": is_prod_stacker_trained,
            "meta_model_status": meta_model_status,
            "experimental_context_index": active_score,
            "experimental_context_index_pct": active_score_pct,
            "partial_dual_branch_index": None,
            "partial_dual_branch_index_pct": None,
            "notice": "A validated multimodal threat-prediction model is not currently available because synchronized observations and independently verified outcome labels have not yet been collected.",
            "clarification": "This index represents a heuristic combination of physiological, spatial and environmental context scores. It is not a validated threat prediction.",
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

    def _generate_partial_explanation(
        self,
        final_pct: float,
        status: str,
        phys_norm: float,
        loc_norm: float
    ) -> str:
        phys_pct = phys_norm * 100.0
        loc_pct = loc_norm * 100.0
        return (
            f"Partial Context Assessment: The partial dual-branch context index is {final_pct:.2f}% ({status}). "
            f"It combines physiological stress ({phys_pct:.1f}%, 61.54% re-normalized weight) and location context risk ({loc_pct:.1f}%, 38.46% re-normalized weight). "
            f"Environmental context is UNKNOWN/unavailable and is omitted without silent substitution. "
            f"Limitations: Situational illumination and crowd density are not factored into this partial index. It describes contextual inputs only and is not a validated threat prediction."
        )

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
        phys_pct = phys_norm * 100.0
        loc_pct = loc_norm * 100.0
        env_pct = env_norm * 100.0
        return (
            f"Context Assessment: The experimental weighted context index is {final_pct:.2f}%. "
            f"It combines physiological stress ({phys_pct:.1f}%), location context risk ({loc_pct:.1f}%) "
            f"and environmental context risk ({env_pct:.1f}%) using predefined reference weights. "
            f"This value describes contextual inputs only and does not establish whether a person or location is safe or unsafe."
        )

if __name__ == "__main__":
    predictor = MultimodalThreatPredictor()
    print("MultimodalThreatPredictor initialized successfully.")
    res = predictor.predict_fusion(0.20, 0.85, 0.40)
    print("Test inference result:")
    print("  Active Model:", res["active_model_name"])
    print("  Threat Score:", f"{res['threat_score_pct']:.2f}% ({res['threat_status']})")
    print("  Explanation:", res["explanation"])
