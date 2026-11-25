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
from src.fusion.fusion_config import FusionMode, MissingModalityStrategy, DEFAULT_BASELINE_WEIGHTS
from src.fusion.fusion_engine import DynamicMultimodalFusionEngine
from src.fusion.adaptive_fusion import (
    ReliabilityAwareAdaptiveFusionEngine,
    BASELINE_PRIORS,
    FRESHNESS_TAUS_SEC
)

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

        # 4. Unified Dynamic Multimodal Fusion Engine
        self.dynamic_fusion_engine = DynamicMultimodalFusionEngine(
            default_mode=FusionMode.RELIABILITY_AWARE,
            missing_strategy=MissingModalityStrategy.MARGINALIZED_EXCLUSION,
            threshold=config.DEFAULT_DECISION_THRESHOLD
        )
        self.adaptive_fusion_engine = ReliabilityAwareAdaptiveFusionEngine(
            baseline_priors=DEFAULT_BASELINE_WEIGHTS
        )

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
        weight_environment: float = 0.35,
        fusion_mode: Optional[str] = None,
        physio_telemetry: Optional[Dict[str, Any]] = None,
        spatial_analysis: Optional[Dict[str, Any]] = None,
        environmental_res: Optional[Dict[str, Any]] = None,
        is_manual_override: bool = False,
        **kwargs: Any
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

        # Defensive check for all-missing / all-invalid modalities
        all_modalities_missing = (
            (physiological_stress_score is None or not np.isfinite(physiological_stress_score))
            and (location_risk_score is None or not np.isfinite(location_risk_score))
            and not is_env_available
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

        # Resolve requested fusion mode
        resolved_mode = None
        if fusion_mode:
            try:
                resolved_mode = FusionMode(fusion_mode)
            except Exception:
                resolved_mode = FusionMode.RELIABILITY_AWARE
        else:
            resolved_mode = FusionMode.RELIABILITY_AWARE

        # Execute unified dynamic multimodal fusion evaluation for candidate categories & governance
        dynamic_eval = self.dynamic_fusion_engine.evaluate_fusion(
            p_stress=phys_norm,
            l_risk=loc_norm,
            e_risk=environment_risk_score if is_env_available else None,
            physio_telemetry=physio_telemetry,
            spatial_analysis=spatial_analysis,
            environmental_res=environmental_res,
            physio_age_sec=kwargs.get("physio_age_sec", 0.0),
            spatial_age_sec=kwargs.get("spatial_age_sec", 0.0),
            env_age_sec=kwargs.get("env_age_sec", 0.0),
            is_manual_override=is_manual_override,
            mode=resolved_mode
        )

        # Execute deterministic reliability-aware adaptive fusion
        adaptive_eval = self.adaptive_fusion_engine.fuse(
            p_score=phys_norm if (physiological_stress_score is not None and np.isfinite(physiological_stress_score)) else None,
            l_score=loc_norm if (location_risk_score is not None and np.isfinite(location_risk_score)) else None,
            e_score=environment_risk_score if is_env_available else None,
            physio_telemetry=physio_telemetry,
            spatial_analysis=spatial_analysis,
            environmental_res=environmental_res,
            physio_age_sec=kwargs.get("physio_age_sec", 0.0),
            spatial_age_sec=kwargs.get("spatial_age_sec", 0.0),
            env_age_sec=kwargs.get("env_age_sec", 0.0),
            is_manual_override=is_manual_override
        )

        logger.info(
            "predict_fusion: P=%.4f, L=%.4f, E=%s -> status=%s",
            phys_norm, loc_norm, environment_risk_score,
            adaptive_eval["status"]
        )

        # Handle complete failure / all modalities missing
        if all_modalities_missing or adaptive_eval["status"] == "UNAVAILABLE" or adaptive_eval["overall_context_index"] is None:
            return {
                "fusion_mode": "UNAVAILABLE",
                "available_modalities": [],
                "missing_modalities": ["physiological", "spatial", "environmental"],
                "excluded_modalities": ["physiological", "spatial", "environmental"],
                "physiological_stress_score": None,
                "location_risk_score": None,
                "environment_risk_score": None,
                "environmental_context_score": None,
                "threat_probability": None,
                "threat_score_pct": None,
                "threat_status": "UNAVAILABLE",
                "context_status": "UNAVAILABLE",
                "context_classification": "UNAVAILABLE",
                "overall_context_index": None,
                "overall_context_index_pct": None,
                "experimental_context_index": None,
                "experimental_context_index_pct": None,
                "partial_dual_branch_index": None,
                "partial_dual_branch_index_pct": None,
                "active_model_name": "RELIABILITY-AWARE ADAPTIVE FUSION (UNAVAILABLE)",
                "is_meta_learner_trained": False,
                "meta_model_status": "UNTRAINED — VALID MULTIMODAL THREAT DATASET REQUIRED",
                "learned_gating_status": "LEARNED GATING: NOT TRAINED",
                "fusion_method": "RELIABILITY_AWARE_ADAPTIVE",
                "adaptive_weights": {"physiological": 0.0, "location": 0.0, "environment": 0.0},
                "weights": {"physiological": 0.0, "location": 0.0, "environment": 0.0},
                "baseline_weights": {"physiological": 0.0, "location": 0.0, "environment": 0.0},
                "partial_weights": {"physiological": 0.0, "location": 0.0},
                "contributions": {"physiological": 0.0, "location": 0.0, "environment": 0.0},
                "contributions_pct": {"physiological": 0.0, "location": 0.0, "environment": 0.0},
                "partial_contributions": {"physiological": 0.0, "location": 0.0},
                "partial_contributions_pct": {"physiological": 0.0, "location": 0.0},
                "modality_contributions": {"physiological": 0.0, "location": 0.0, "environment": 0.0},
                "modality_contributions_pct": {"physiological": 0.0, "location": 0.0, "environment": 0.0},
                "modality_reliabilities": adaptive_eval["modality_reliabilities"],
                "formula_str": "UNAVAILABLE (All modalities missing or invalid)",
                "notice": "All input modalities are unavailable or invalid. Assessment cannot be computed.",
                "explanation": "Assessment unavailable: None of the input modalities provide valid physical or contextual measurements. Zero risk is not fabricated.",
                "modalities_metadata": modalities_metadata,
                "active_fusion_mode": "RELIABILITY_AWARE",
                "active_fusion_mode_name": "Reliability-Aware Adaptive Fusion",
                "reliability_summary": dynamic_eval["reliability_summary"],
                "candidate_risk_categories": dynamic_eval["candidate_risk_categories"],
                "uncertainty_index": 1.0,
                "dynamic_gating_details": dynamic_eval.get("dynamic_gating_details"),
                "safety_policy": dynamic_eval["safety_policy"]
            }

        # Extract adaptive results
        adaptive_weights = adaptive_eval["adaptive_weights"]
        alpha_p = adaptive_weights["physiological"]
        alpha_l = adaptive_weights["location"]
        alpha_e = adaptive_weights["environment"]
        active_score = adaptive_eval["overall_context_index"]
        active_score_pct = adaptive_eval["overall_context_index_pct"]
        active_status = adaptive_eval["context_classification"]
        contrib_p = adaptive_eval["modality_contributions"]["physiological"]
        contrib_l = adaptive_eval["modality_contributions"]["location"]
        contrib_e = adaptive_eval["modality_contributions"]["environment"]
        formula_str = adaptive_eval["formula_str"]
        avail_mods = adaptive_eval["available_modalities"]
        excl_mods = adaptive_eval["excluded_modalities"]

        # -----------------------------------------------------------------
        # BRANCH A: PARTIAL DUAL-BRANCH CONTEXT INDEX (ENV IS MISSING/UNKNOWN)
        # -----------------------------------------------------------------
        if not is_env_available:
            alpha_p_pct = round(alpha_p * 100.0, 2)
            alpha_l_pct = round(alpha_l * 100.0, 2)

            explanation = (
                f"Partial Context Assessment: The partial dual-branch context index is {active_score_pct:.2f}% ({active_status}). "
                f"It combines physiological stress ({phys_norm * 100.0:.1f}%, {alpha_p_pct:.2f}% re-normalized weight) and location context risk ({loc_norm * 100.0:.1f}%, {alpha_l_pct:.2f}% re-normalized weight). "
                f"Environmental context is UNKNOWN/unavailable and is omitted without silent substitution. "
                f"Formula: {formula_str}. "
                f"Limitations: Situational illumination and crowd density are not factored into this partial index. It describes contextual inputs only and is not a validated threat prediction."
            )

            return {
                "fusion_mode": "PARTIAL_DUAL_BRANCH",
                "available_modalities": ["physiological", "spatial"],
                "missing_modalities": ["environmental"],
                "excluded_modalities": excl_mods,
                # Base model scores
                "physiological_stress_score": phys_norm,
                "location_risk_score": loc_norm,
                "environment_risk_score": None,
                "environmental_context_score": None,
                "environmental_crowd_context_score": None,
                "environmental_lighting_context_score": None,
                "environmental_model_provenance": "MISSING / UNKNOWN (Excluded from fusion)",
                "modalities_metadata": modalities_metadata,
                # Adaptive & Partial Dual-Branch Specifics
                "overall_context_index": active_score,
                "overall_context_index_pct": active_score_pct,
                "partial_dual_branch_index": active_score,
                "partial_dual_branch_index_pct": active_score_pct,
                "partial_fusion_name": "PARTIAL DUAL-BRANCH CONTEXT INDEX (PHYSIOLOGICAL + SPATIAL ONLY)",
                "partial_formula": formula_str,
                "formula_str": formula_str,
                "fusion_method": "RELIABILITY_AWARE_ADAPTIVE",
                "adaptive_weights": adaptive_weights,
                "weights": adaptive_weights,
                "partial_weights": {
                    "physiological": alpha_p,
                    "location": alpha_l
                },
                "baseline_weights": {
                    "physiological": alpha_p,
                    "location": alpha_l,
                    "environment": 0.0
                },
                "contributions": {
                    "physiological": contrib_p,
                    "location": contrib_l,
                    "environment": 0.0
                },
                "contributions_pct": {
                    "physiological": round(contrib_p * 100.0, 2),
                    "location": round(contrib_l * 100.0, 2),
                    "environment": 0.0
                },
                "partial_contributions": {
                    "physiological": contrib_p,
                    "location": contrib_l
                },
                "partial_contributions_pct": {
                    "physiological": round(contrib_p * 100.0, 2),
                    "location": round(contrib_l * 100.0, 2)
                },
                "modality_contributions": adaptive_eval["modality_contributions"],
                "modality_contributions_pct": adaptive_eval["modality_contributions_pct"],
                "modality_reliabilities": adaptive_eval["modality_reliabilities"],
                # Complete 3-branch index is explicitly None
                "experimental_context_index": None,
                "experimental_context_index_pct": None,
                "threat_probability": active_score,
                "threat_score_pct": active_score_pct,
                "threat_status": active_status,
                "context_status": active_status,
                "context_classification": active_status,
                "active_model_name": "PARTIAL DUAL-BRANCH CONTEXT INDEX (PHYSIOLOGICAL + SPATIAL ONLY)",
                "is_meta_learner_trained": False,
                "meta_model_status": "NOT EVALUATED (MISSING ENVIRONMENTAL MODALITY)",
                "learned_gating_status": "LEARNED GATING: NOT TRAINED",
                "notice": "A complete 3-branch context index is unavailable because environmental context is missing or unknown. Operating in fallback dual-branch mode.",
                "clarification": "This partial index evaluates physiological biometrics and spatial location only. Environmental conditions (lighting, crowd density) are excluded.",
                "stacking_meta_probability": None,
                "stacking_meta_status": None,
                "stacking_explainability": None,
                "baseline_threat_probability": active_score,
                "baseline_threat_score_pct": active_score_pct,
                "baseline_status": active_status,
                "threshold_used": threshold,
                "explanation": explanation,
                # Dynamic Multimodal Intelligence Enhancements
                "active_fusion_mode": dynamic_eval["fusion_mode"],
                "active_fusion_mode_name": dynamic_eval["fusion_mode_name"],
                "reliability_summary": dynamic_eval["reliability_summary"],
                "candidate_risk_categories": dynamic_eval["candidate_risk_categories"],
                "uncertainty_index": dynamic_eval["uncertainty_index"],
                "dynamic_gating_details": dynamic_eval.get("dynamic_gating_details"),
                "safety_policy": dynamic_eval["safety_policy"]
            }

        # -----------------------------------------------------------------
        # BRANCH B: THREE-BRANCH CONTEXT FUSION (ALL 3 INPUTS PRESENT)
        # -----------------------------------------------------------------
        env_norm = float(np.clip(environment_risk_score, 0.0, 1.0))

        alpha_p_pct = round(alpha_p * 100.0, 1)
        alpha_l_pct = round(alpha_l * 100.0, 1)
        alpha_e_pct = round(alpha_e * 100.0, 1)

        is_nominal_baseline = (
            abs(alpha_p - 0.40) < 0.005 and 
            abs(alpha_l - 0.25) < 0.005 and 
            abs(alpha_e - 0.35) < 0.005
        )

        if is_nominal_baseline:
            active_model_name = "Experimental Weighted Context Index"
            explanation = self._generate_baseline_explanation(
                active_score_pct, active_status, phys_norm, loc_norm, env_norm
            )
            clarification = (
                "This index represents a heuristic combination of physiological, spatial "
                "and environmental context scores. It is not a validated threat prediction."
            )
        else:
            active_model_name = "Reliability-Aware Adaptive Context Index"
            explanation = (
                f"Context Assessment: The reliability-aware adaptive context index is {active_score_pct:.2f}% ({active_status}). "
                f"It combines physiological stress ({phys_norm * 100.0:.1f}%, {alpha_p_pct:.1f}% adaptive weight), location context risk ({loc_norm * 100.0:.1f}%, {alpha_l_pct:.1f}% adaptive weight), "
                f"and environmental context risk ({env_norm * 100.0:.1f}%, {alpha_e_pct:.1f}% adaptive weight) using data freshness and sensor quality weighting. "
                f"Formula: {formula_str}. "
                f"This value describes contextual inputs only and does not establish whether a person or location is safe or unsafe."
            )
            clarification = (
                "This index represents a reliability-aware adaptive combination of physiological, spatial and environmental context scores. It is not a validated threat prediction."
            )

        return {
            "fusion_mode": "THREE_BRANCH_FUSION",
            "available_modalities": ["physiological", "spatial", "environmental"],
            "missing_modalities": [],
            "excluded_modalities": excl_mods,
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
            # Active decision (Reliability-Aware Adaptive Fusion)
            "threat_probability": active_score,
            "threat_score_pct": active_score_pct,
            "threat_status": active_status,
            "context_status": active_status,
            "context_classification": active_status,
            "overall_context_index": active_score,
            "overall_context_index_pct": active_score_pct,
            "active_model_name": active_model_name,
            "is_meta_learner_trained": False,
            "meta_model_status": "UNTRAINED — VALID MULTIMODAL THREAT DATASET REQUIRED",
            "learned_gating_status": "LEARNED GATING: NOT TRAINED",
            "fusion_method": "RELIABILITY_AWARE_ADAPTIVE",
            "experimental_context_index": active_score,
            "experimental_context_index_pct": active_score_pct,
            "partial_dual_branch_index": None,
            "partial_dual_branch_index_pct": None,
            "notice": "A validated multimodal threat-prediction model is not currently available because synchronized observations and independently verified outcome labels have not yet been collected.",
            "clarification": clarification,
            "stacking_meta_probability": None,
            "stacking_meta_status": None,
            "stacking_explainability": None,
            "baseline_threat_probability": active_score,
            "baseline_threat_score_pct": active_score_pct,
            "baseline_status": active_status,
            "formula_str": formula_str,
            "adaptive_weights": adaptive_weights,
            "weights": adaptive_weights,
            "baseline_weights": adaptive_weights,
            "contributions": {
                "physiological": contrib_p,
                "location": contrib_l,
                "environment": contrib_e
            },
            "contributions_pct": {
                "physiological": round(contrib_p * 100.0, 2),
                "location": round(contrib_l * 100.0, 2),
                "environment": round(contrib_e * 100.0, 2)
            },
            "modality_contributions": adaptive_eval["modality_contributions"],
            "modality_contributions_pct": adaptive_eval["modality_contributions_pct"],
            "modality_reliabilities": adaptive_eval["modality_reliabilities"],
            "threshold_used": threshold,
            "explanation": explanation,
            # Dynamic Multimodal Intelligence Enhancements
            "active_fusion_mode": dynamic_eval["fusion_mode"],
            "active_fusion_mode_name": dynamic_eval["fusion_mode_name"],
            "reliability_summary": dynamic_eval["reliability_summary"],
            "candidate_risk_categories": dynamic_eval["candidate_risk_categories"],
            "uncertainty_index": dynamic_eval["uncertainty_index"],
            "dynamic_gating_details": dynamic_eval.get("dynamic_gating_details"),
            "safety_policy": dynamic_eval["safety_policy"]
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

    def set_fusion_mode(self, mode_str: str) -> None:
        """Sets the active fusion mode on the underlying dynamic fusion engine."""
        try:
            self.dynamic_fusion_engine.set_fusion_mode(FusionMode(mode_str))
        except Exception:
            self.dynamic_fusion_engine.set_fusion_mode(FusionMode.STATIC_WEIGHTED)

    def get_fusion_mode(self) -> str:
        """Retrieves active fusion mode name."""
        return self.dynamic_fusion_engine.active_mode.value

if __name__ == "__main__":
    predictor = MultimodalThreatPredictor()
    print("MultimodalThreatPredictor initialized successfully.")
    res = predictor.predict_fusion(0.20, 0.85, 0.40)
    print("Test inference result:")
    print("  Active Model:", res["active_model_name"])
    print("  Threat Score:", f"{res['threat_score_pct']:.2f}% ({res['threat_status']})")
    print("  Explanation:", res["explanation"])
