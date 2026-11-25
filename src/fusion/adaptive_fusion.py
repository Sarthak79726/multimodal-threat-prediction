"""
Reliability-Aware Adaptive Multimodal Fusion Engine.

Implements mathematically grounded, deterministic, reliability-aware adaptive fusion:
    q_i = validity_i * freshness_i * source_reliability_i
    freshness_i = exp(-age_i / tau_i)
    u_i = b_i * q_i
    alpha_i = u_i / sum(u_j)
    R = alpha_P * P + alpha_L * L + alpha_E * E

Where:
    - P, L, E are the Level 0 modality scores for Physiological, Location, and Environmental branches.
    - b = [0.40, 0.25, 0.35] are baseline reference priors.
    - tau_i is a modality-specific freshness decay timescale.
    - alpha_i are dynamically adapted, normalized weights summing to 1.0 when at least one modality is valid.
    - Missing or invalid modalities receive zero weight (u_i = 0, alpha_i = 0).
    - If all modalities are invalid, returns an UNAVAILABLE result with score=None (never silent zero risk).
    - Transparently reports learned gating training status as NOT TRAINED until verified real-world labels exist.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
import math
import numpy as np
import logging

logger = logging.getLogger(__name__)

# Modality baseline priors (Operational Reference)
BASELINE_PRIORS: Dict[str, float] = {
    "physiological": 0.40,
    "location": 0.25,
    "environment": 0.35
}

# Modality freshness decay timescales tau_i (in seconds)
FRESHNESS_TAUS_SEC: Dict[str, float] = {
    "physiological": 120.0,   # Biometric signals: HR, EDA decay rapidly
    "spatial": 300.0,         # Location / GPS coordinates: moderate movement timescale
    "environment": 900.0      # Environmental context: weather, crowd, lighting schedule
}

# Context classification thresholds
CONTEXT_THRESHOLDS: Dict[str, float] = {
    "LOW_MAX": 0.39,      # Standard Context
    "MEDIUM_MAX": 0.69,   # Elevated Context
}                         # > 0.69: High Context


@dataclass
class ModalityReliabilityMetrics:
    """Detailed reliability metrics for a single modality stream."""
    modality: str
    validity: float                   # in [0.0, 1.0]
    freshness: float                  # in [0.0, 1.0] from exp(-age / tau)
    source_reliability: float         # in [0.0, 1.0] based on physical sensor / source quality
    composite_q: float                # q_i = validity * freshness * source_reliability in [0.0, 1.0]
    age_seconds: float
    tau_seconds: float
    status: str                       # OPTIMAL, ACCEPTABLE, DEGRADED, STALE, UNAVAILABLE
    is_valid: bool
    is_stale: bool
    source_description: str
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "modality": self.modality,
            "validity": round(self.validity, 4),
            "freshness": round(self.freshness, 4),
            "source_reliability": round(self.source_reliability, 4),
            "composite_q": round(self.composite_q, 4),
            "age_seconds": round(self.age_seconds, 1),
            "tau_seconds": round(self.tau_seconds, 1),
            "status": self.status,
            "is_valid": self.is_valid,
            "is_stale": self.is_stale,
            "source_description": self.source_description,
            "warnings": self.warnings
        }


class ReliabilityAwareAdaptiveFusionEngine:
    """
    Deterministic, reliability-aware adaptive fusion engine.
    Adapts modality contributions based on measured physical validity, temporal freshness,
    and sensor provenance without fabricated heuristics or arbitrary situation multipliers.
    """

    def __init__(
        self,
        baseline_priors: Optional[Dict[str, float]] = None,
        freshness_taus: Optional[Dict[str, float]] = None
    ):
        self.baseline_priors = dict(baseline_priors or BASELINE_PRIORS)
        self.freshness_taus = dict(freshness_taus or FRESHNESS_TAUS_SEC)

    def evaluate_physiological_reliability(
        self,
        score: Optional[float],
        telemetry: Optional[Dict[str, Any]] = None,
        age_seconds: float = 0.0,
        tau_seconds: Optional[float] = None
    ) -> ModalityReliabilityMetrics:
        """Evaluates physiological biometric stream reliability."""
        tau = float(tau_seconds or self.freshness_taus["physiological"])
        age = max(0.0, float(age_seconds))
        warnings: List[str] = []

        # 1. Validity
        is_finite_score = (score is not None and isinstance(score, (int, float)) and np.isfinite(score) and 0.0 <= float(score) <= 1.0)
        if not is_finite_score:
            warnings.append("Physiological stress score is missing or invalid.")
            validity = 0.0
        else:
            validity = 1.0
            if telemetry and isinstance(telemetry, dict):
                # Channel completeness check
                eda = telemetry.get("eda_mean")
                temp = telemetry.get("temp_mean")
                ecg = telemetry.get("ecg_std")
                hr = telemetry.get("heart_rate_bpm")
                channels = [eda is not None, temp is not None, ecg is not None, hr is not None]
                if any(channels):
                    completeness = sum(channels) / float(len(channels))
                    validity *= completeness
                    if completeness < 1.0:
                        warnings.append(f"Incomplete biometric channels ({sum(channels)}/4 active).")

                # Plausibility bounds
                if hr is not None and (hr < 40 or hr > 200):
                    validity *= 0.70
                    warnings.append(f"Heart rate {hr} BPM outside physiological baseline [40, 200].")
                if temp is not None and (temp < 20.0 or temp > 43.0):
                    validity *= 0.70
                    warnings.append(f"Skin temperature {temp}°C outside plausible physiological range.")

        validity = float(np.clip(validity, 0.0, 1.0))
        is_valid = (validity > 0.0)

        # 2. Freshness: exp(-age / tau)
        freshness = float(math.exp(-age / tau)) if tau > 0 else 1.0
        freshness = float(np.clip(freshness, 0.0, 1.0))
        is_stale = (age > tau)
        if is_stale:
            warnings.append(f"Biometric data age ({age:.1f}s) exceeds freshness timescale ({tau:.1f}s).")

        # 3. Source Reliability
        source_desc = "WESAD Research Benchmark (Verified Laboratory Ground Truth)"
        if not is_valid:
            source_rel = 0.0
            source_desc = "No valid biometric stream available"
        elif telemetry and isinstance(telemetry, dict):
            conn = telemetry.get("connection_status", "CONNECTED")
            if conn == "DISCONNECTED":
                source_rel = 0.0
                source_desc = "Wearable disconnected (Awaiting live sensor)"
            elif telemetry.get("is_live_sensor"):
                source_rel = 1.0  # Live authenticated sensor
                source_desc = "Live Wearable Sensor Stream (Authenticated)"
            else:
                source_rel = 1.0  # Laboratory benchmark (WESAD replay)
                source_desc = "WESAD Benchmark Laboratory Telemetry"
        else:
            source_rel = 1.0

        source_rel = float(np.clip(source_rel, 0.0, 1.0))

        # Composite q_i
        q_p = float(validity * freshness * source_rel) if is_valid else 0.0
        q_p = float(np.clip(q_p, 0.0, 1.0))

        if not is_valid:
            status = "UNAVAILABLE"
        elif is_stale or q_p < 0.30:
            status = "STALE" if is_stale else "DEGRADED"
        elif q_p >= 0.75:
            status = "OPTIMAL"
        else:
            status = "ACCEPTABLE"

        return ModalityReliabilityMetrics(
            modality="physiological",
            validity=validity,
            freshness=freshness,
            source_reliability=source_rel,
            composite_q=q_p,
            age_seconds=age,
            tau_seconds=tau,
            status=status,
            is_valid=is_valid,
            is_stale=is_stale,
            source_description=source_desc,
            warnings=warnings
        )

    def evaluate_spatial_reliability(
        self,
        score: Optional[float],
        spatial_analysis: Optional[Dict[str, Any]] = None,
        age_seconds: float = 0.0,
        tau_seconds: Optional[float] = None,
        is_manual_override: bool = False
    ) -> ModalityReliabilityMetrics:
        """Evaluates spatial GPS / campus landmark stream reliability."""
        tau = float(tau_seconds or self.freshness_taus["spatial"])
        age = max(0.0, float(age_seconds))
        warnings: List[str] = []

        # 1. Validity
        is_finite_score = (score is not None and isinstance(score, (int, float)) and np.isfinite(score) and 0.0 <= float(score) <= 1.0)
        if not is_finite_score:
            warnings.append("Location risk score is missing or invalid.")
            validity = 0.0
        else:
            validity = 1.0
            if spatial_analysis and isinstance(spatial_analysis, dict):
                st_str = spatial_analysis.get("status", "CONNECTED")
                if st_str == "GPS_UNAVAILABLE":
                    validity = 0.0 if not is_manual_override else 1.0
                    if not is_manual_override:
                        warnings.append("GPS positioning is unavailable.")
                elif st_str == "OUTSIDE_CAMPUS_GEOFENCE" and not is_manual_override:
                    validity *= 0.50
                    warnings.append("Coordinates fall outside BIT Mesra campus geofence.")

        validity = float(np.clip(validity, 0.0, 1.0))
        is_valid = (validity > 0.0)

        # 2. Freshness: exp(-age / tau)
        # If manual override was applied, user explicitly confirmed landmark for this evaluation
        if is_manual_override:
            freshness = 1.0
            is_stale = False
        else:
            freshness = float(math.exp(-age / tau)) if tau > 0 else 1.0
            freshness = float(np.clip(freshness, 0.0, 1.0))
            is_stale = (age > tau)
            if is_stale:
                warnings.append(f"Location data age ({age:.1f}s) exceeds freshness timescale ({tau:.1f}s).")

        # 3. Source Reliability
        if not is_valid:
            source_rel = 0.0
            source_desc = "No valid location stream"
        elif is_manual_override:
            source_rel = 1.0  # User manually verified landmark location
            source_desc = "Manual Landmark Selection (User-Verified Ground Truth)"
        elif spatial_analysis and isinstance(spatial_analysis, dict):
            c_loc = spatial_analysis.get("current_location", {})
            acc = c_loc.get("accuracy_meters") if isinstance(c_loc, dict) else None
            st_str = spatial_analysis.get("status", "CONNECTED")

            if acc is not None:
                if acc <= 10.0:
                    source_rel = 1.0
                elif acc <= 30.0:
                    source_rel = 0.90
                elif acc <= 100.0:
                    source_rel = max(0.50, 1.0 - 0.5 * ((acc - 10.0) / 90.0))
                else:
                    source_rel = max(0.20, 50.0 / acc)
                    warnings.append(f"Low GPS accuracy radius (±{acc:.1f}m).")
            else:
                source_rel = 1.0

            if st_str == "AMBIGUOUS_LOCATION":
                source_rel *= 0.75
                warnings.append("GPS location equidistant between multiple campus landmarks.")

            source_desc = f"Browser Geolocation API (±{acc:.1f}m)" if acc else "BIT Mesra Geofence Engine"
        else:
            source_rel = 1.0
            source_desc = "BIT Mesra Geofence Engine"

        source_rel = float(np.clip(source_rel, 0.0, 1.0))

        # Composite q_i
        q_l = float(validity * freshness * source_rel) if is_valid else 0.0
        q_l = float(np.clip(q_l, 0.0, 1.0))

        if not is_valid:
            status = "UNAVAILABLE"
        elif is_stale or q_l < 0.30:
            status = "STALE" if is_stale else "DEGRADED"
        elif q_l >= 0.75:
            status = "OPTIMAL"
        else:
            status = "ACCEPTABLE"

        return ModalityReliabilityMetrics(
            modality="spatial",
            validity=validity,
            freshness=freshness,
            source_reliability=source_rel,
            composite_q=q_l,
            age_seconds=age,
            tau_seconds=tau,
            status=status,
            is_valid=is_valid,
            is_stale=is_stale,
            source_description=source_desc,
            warnings=warnings
        )

    def evaluate_environmental_reliability(
        self,
        score: Optional[float],
        environmental_res: Optional[Dict[str, Any]] = None,
        age_seconds: float = 0.0,
        tau_seconds: Optional[float] = None
    ) -> ModalityReliabilityMetrics:
        """Evaluates environmental stream (weather API + crowd/lighting context) reliability."""
        tau = float(tau_seconds or self.freshness_taus["environment"])
        age = max(0.0, float(age_seconds))
        warnings: List[str] = []

        # 1. Validity
        is_finite_score = (score is not None and isinstance(score, (int, float)) and np.isfinite(score) and 0.0 <= float(score) <= 1.0)
        if not is_finite_score:
            warnings.append("Environmental context score is missing or unknown.")
            validity = 0.0
        elif environmental_res and isinstance(environmental_res, dict) and environmental_res.get("status") not in (None, "SUCCESS"):
            warnings.append(f"Environmental inference flagged: {environmental_res.get('status')}.")
            validity = 0.0
        else:
            validity = 1.0

        validity = float(np.clip(validity, 0.0, 1.0))
        is_valid = (validity > 0.0)

        # 2. Freshness: exp(-age / tau)
        freshness = float(math.exp(-age / tau)) if tau > 0 else 1.0
        freshness = float(np.clip(freshness, 0.0, 1.0))
        is_stale = (age > tau)
        if is_stale:
            warnings.append(f"Environmental data age ({age:.1f}s) exceeds freshness timescale ({tau:.1f}s).")

        # 3. Source Reliability
        if not is_valid:
            source_rel = 0.0
            source_desc = "Environmental stream unavailable"
        elif environmental_res and isinstance(environmental_res, dict):
            # Check for physical IoT hardware vs verified Live API vs Schedule Model
            crowd_res = environmental_res.get("crowd_prediction", {})
            light_res = environmental_res.get("lighting_prediction", {})
            c_conf = (crowd_res.get("confidence_pct", 100.0) / 100.0) if (isinstance(crowd_res, dict) and crowd_res.get("confidence_pct") is not None) else 1.0
            l_conf = (light_res.get("confidence_pct", 100.0) / 100.0) if (isinstance(light_res, dict) and light_res.get("confidence_pct") is not None) else 1.0

            # Verified Open-Meteo API provides high atmospheric authenticity
            has_live_weather = bool(environmental_res.get("weather_reading") and environmental_res["weather_reading"].get("status") == "LIVE")
            auth_factor = 1.0 if has_live_weather else 0.95

            source_rel = float(0.40 * auth_factor + 0.30 * c_conf + 0.30 * l_conf)
            source_desc = "Open-Meteo Live API + Campus AI Context Models (XGBoost + Random Forest)"
        else:
            source_rel = 1.0
            source_desc = "Campus Context Model"

        source_rel = float(np.clip(source_rel, 0.0, 1.0))

        # Composite q_i
        q_e = float(validity * freshness * source_rel) if is_valid else 0.0
        q_e = float(np.clip(q_e, 0.0, 1.0))

        if not is_valid:
            status = "UNAVAILABLE"
        elif is_stale or q_e < 0.30:
            status = "STALE" if is_stale else "DEGRADED"
        elif q_e >= 0.75:
            status = "OPTIMAL"
        else:
            status = "ACCEPTABLE"

        return ModalityReliabilityMetrics(
            modality="environment",
            validity=validity,
            freshness=freshness,
            source_reliability=source_rel,
            composite_q=q_e,
            age_seconds=age,
            tau_seconds=tau,
            status=status,
            is_valid=is_valid,
            is_stale=is_stale,
            source_description=source_desc,
            warnings=warnings
        )

    def fuse(
        self,
        p_score: Optional[float],
        l_score: Optional[float],
        e_score: Optional[float],
        physio_telemetry: Optional[Dict[str, Any]] = None,
        spatial_analysis: Optional[Dict[str, Any]] = None,
        environmental_res: Optional[Dict[str, Any]] = None,
        physio_age_sec: float = 0.0,
        spatial_age_sec: float = 0.0,
        env_age_sec: float = 0.0,
        is_manual_override: bool = False
    ) -> Dict[str, Any]:
        """
        Executes reliability-aware adaptive fusion:
        1. Evaluates q_i = validity_i * freshness_i * source_reliability_i for each modality.
        2. Computes unnormalized contributions u_i = b_i * q_i using baseline priors b = [0.40, 0.25, 0.35].
        3. Normalizes alpha_i = u_i / sum(u_j).
        4. Calculates R = alpha_P * P + alpha_L * L + alpha_E * E.
        """
        # 1. Modality Reliability Reports
        rel_p = self.evaluate_physiological_reliability(
            score=p_score,
            telemetry=physio_telemetry,
            age_seconds=physio_age_sec
        )
        rel_l = self.evaluate_spatial_reliability(
            score=l_score,
            spatial_analysis=spatial_analysis,
            age_seconds=spatial_age_sec,
            is_manual_override=is_manual_override
        )
        rel_e = self.evaluate_environmental_reliability(
            score=e_score,
            environmental_res=environmental_res,
            age_seconds=env_age_sec
        )

        q_p = rel_p.composite_q
        q_l = rel_l.composite_q
        q_e = rel_e.composite_q

        # Baseline priors b_i
        b_p = self.baseline_priors.get("physiological", 0.40)
        b_l = self.baseline_priors.get("location", 0.25)
        b_e = self.baseline_priors.get("environment", 0.35)

        # 2. Unnormalized contributions u_i = b_i * q_i
        u_p = b_p * q_p
        u_l = b_l * q_l
        u_e = b_e * q_e

        u_sum = u_p + u_l + u_e

        available_modalities = []
        if rel_p.is_valid and q_p > 0.0:
            available_modalities.append("physiological")
        if rel_l.is_valid and q_l > 0.0:
            available_modalities.append("location")
        if rel_e.is_valid and q_e > 0.0:
            available_modalities.append("environment")

        excluded_modalities = []
        if "physiological" not in available_modalities:
            excluded_modalities.append("physiological")
        if "location" not in available_modalities:
            excluded_modalities.append("location")
        if "environment" not in available_modalities:
            excluded_modalities.append("environment")

        # 3. Handle case where all modalities are invalid / missing
        if u_sum <= 0.0 or len(available_modalities) == 0:
            logger.warning("Adaptive fusion failed: All modalities are invalid or missing.")
            return {
                "status": "UNAVAILABLE",
                "final_context_score": None,
                "final_context_score_pct": None,
                "overall_context_index": None,
                "overall_context_index_pct": None,
                "context_classification": "UNAVAILABLE",
                "fusion_mode": "UNAVAILABLE",
                "fusion_method": "RELIABILITY_AWARE_ADAPTIVE",
                "is_available": False,
                "message": "All modalities are invalid, missing, or have zero reliability. Context index is UNAVAILABLE.",
                "modality_scores": {
                    "physiological": None,
                    "location": None,
                    "spatial": None,
                    "environment": None,
                    "environmental": None
                },
                "modality_reliability": {
                    "physiological": 0.0,
                    "location": 0.0,
                    "spatial": 0.0,
                    "environment": 0.0,
                    "environmental": 0.0
                },
                "modality_freshness": {
                    "physiological": 0.0,
                    "location": 0.0,
                    "spatial": 0.0,
                    "environment": 0.0,
                    "environmental": 0.0
                },
                "adaptive_weights": {
                    "physiological": 0.0,
                    "location": 0.0,
                    "spatial": 0.0,
                    "environment": 0.0,
                    "environmental": 0.0
                },
                "adaptive_weights_rounded": {
                    "physiological": 0.0,
                    "location": 0.0,
                    "spatial": 0.0,
                    "environment": 0.0,
                    "environmental": 0.0
                },
                "unnormalized_contributions": {
                    "physiological": 0.0,
                    "location": 0.0,
                    "environment": 0.0
                },
                "modality_contributions": {
                    "physiological": 0.0,
                    "location": 0.0,
                    "environment": 0.0
                },
                "modality_contributions_pct": {
                    "physiological": 0.0,
                    "location": 0.0,
                    "environment": 0.0
                },
                "modality_reliabilities": {
                    "physiological": rel_p.to_dict(),
                    "location": rel_l.to_dict(),
                    "environment": rel_e.to_dict()
                },
                "formula_str": "UNAVAILABLE (All modalities missing/invalid)",
                "available_modalities": [],
                "excluded_modalities": ["physiological", "location", "environment"],
                "decision_rationale": "Assessment unavailable: All input modalities (Physiological, Spatial, Environmental) are missing, disconnected, or invalid. Zero risk is not fabricated.",
                "meta_learner_status": "LEARNED GATING: NOT TRAINED",
                "learned_gating_status": "LEARNED GATING: NOT TRAINED"
            }

        # 4. Normalize contributions alpha_i = u_i / sum(u_j)
        alpha_p = u_p / u_sum
        alpha_l = u_l / u_sum
        alpha_e = u_e / u_sum

        # 5. Calculate overall score R = sum(alpha_i * score_i)
        p_val = float(p_score) if (p_score is not None and rel_p.is_valid) else 0.0
        l_val = float(l_score) if (l_score is not None and rel_l.is_valid) else 0.0
        e_val = float(e_score) if (e_score is not None and rel_e.is_valid) else 0.0

        contrib_p = alpha_p * p_val
        contrib_l = alpha_l * l_val
        contrib_e = alpha_e * e_val

        r_score = round(float(contrib_p + contrib_l + contrib_e), 4)
        r_pct = round(r_score * 100.0, 2)

        # Context classification
        if r_score <= CONTEXT_THRESHOLDS["LOW_MAX"]:
            classification = "STANDARD CONTEXT"
        elif r_score <= CONTEXT_THRESHOLDS["MEDIUM_MAX"]:
            classification = "ELEVATED CONTEXT"
        else:
            classification = "HIGH CONTEXT"

        # Build dynamic formula string
        terms = []
        if alpha_p > 0.0:
            terms.append(f"{alpha_p * 100.0:.1f}% × P ({p_val * 100.0:.1f}%)")
        if alpha_l > 0.0:
            terms.append(f"{alpha_l * 100.0:.1f}% × L ({l_val * 100.0:.1f}%)")
        if alpha_e > 0.0:
            terms.append(f"{alpha_e * 100.0:.1f}% × E ({e_val * 100.0:.1f}%)")

        if excluded_modalities:
            ex_str = " [" + ", ".join([m.capitalize() for m in excluded_modalities]) + " Excluded]"
        else:
            ex_str = ""

        formula_str = f"R = {' + '.join(terms)}{ex_str} = {r_pct:.2f}%"

        # Determine explicit fusion mode string
        if len(available_modalities) == 3:
            active_fusion_mode = "THREE_BRANCH_FUSION"
        elif len(available_modalities) == 2:
            active_fusion_mode = "PARTIAL_DUAL_BRANCH"
        else:
            active_fusion_mode = "SINGLE_BRANCH_FUSION"

        # Build dynamic decision rationale from actual backend fusion result
        rationale_parts = []
        if "physiological" in available_modalities:
            rationale_parts.append(f"physiological stress ({p_val * 100.0:.1f}%, {alpha_p * 100.0:.1f}% adaptive weight)")
        if "location" in available_modalities:
            rationale_parts.append(f"location context risk ({l_val * 100.0:.1f}%, {alpha_l * 100.0:.1f}% adaptive weight)")
        if "environment" in available_modalities:
            rationale_parts.append(f"environmental context risk ({e_val * 100.0:.1f}%, {alpha_e * 100.0:.1f}% adaptive weight)")

        comb_desc = " and ".join(rationale_parts) if rationale_parts else "available context streams"
        excl_desc = f" Excluded modalities: {', '.join([m.capitalize() for m in excluded_modalities])} (received 0% weight)." if excluded_modalities else ""

        decision_rationale = (
            f"Context Assessment: The reliability-aware adaptive context index is {r_pct:.2f}% ({classification}). "
            f"It combines {comb_desc} using reliability-aware adaptive fusion with data freshness and sensor quality weighting.{excl_desc} "
            f"Fusion method: {active_fusion_mode}. Formula: {formula_str}. "
            f"This value describes contextual inputs only and does not establish whether a person or location is safe or unsafe."
        )

        return {
            "status": "SUCCESS",
            "final_context_score": r_score,
            "final_context_score_pct": r_pct,
            "overall_context_index": r_score,
            "overall_context_index_pct": r_pct,
            "context_classification": classification,
            "fusion_mode": active_fusion_mode,
            "fusion_method": "RELIABILITY_AWARE_ADAPTIVE",
            "is_available": True,
            "message": "Evaluated via reliability-aware adaptive fusion with data freshness and sensor quality weighting.",
            "modality_scores": {
                "physiological": p_val if rel_p.is_valid else None,
                "location": l_val if rel_l.is_valid else None,
                "spatial": l_val if rel_l.is_valid else None,
                "environment": e_val if rel_e.is_valid else None,
                "environmental": e_val if rel_e.is_valid else None
            },
            "modality_reliability": {
                "physiological": round(q_p, 4),
                "location": round(q_l, 4),
                "spatial": round(q_l, 4),
                "environment": round(q_e, 4),
                "environmental": round(q_e, 4)
            },
            "modality_freshness": {
                "physiological": round(rel_p.freshness, 4),
                "location": round(rel_l.freshness, 4),
                "spatial": round(rel_l.freshness, 4),
                "environment": round(rel_e.freshness, 4),
                "environmental": round(rel_e.freshness, 4)
            },
            "adaptive_weights": {
                "physiological": float(alpha_p),
                "location": float(alpha_l),
                "spatial": float(alpha_l),
                "environment": float(alpha_e),
                "environmental": float(alpha_e)
            },
            "adaptive_weights_rounded": {
                "physiological": round(alpha_p, 4),
                "location": round(alpha_l, 4),
                "spatial": round(alpha_l, 4),
                "environment": round(alpha_e, 4),
                "environmental": round(alpha_e, 4)
            },
            "unnormalized_contributions": {
                "physiological": round(u_p, 4),
                "location": round(u_l, 4),
                "environment": round(u_e, 4)
            },
            "modality_contributions": {
                "physiological": round(contrib_p, 4),
                "location": round(contrib_l, 4),
                "environment": round(contrib_e, 4)
            },
            "modality_contributions_pct": {
                "physiological": round(contrib_p * 100.0, 2),
                "location": round(contrib_l * 100.0, 2),
                "environment": round(contrib_e * 100.0, 2)
            },
            "modality_reliabilities": {
                "physiological": rel_p.to_dict(),
                "location": rel_l.to_dict(),
                "environment": rel_e.to_dict()
            },
            "formula_str": formula_str,
            "available_modalities": available_modalities,
            "excluded_modalities": excluded_modalities,
            "decision_rationale": decision_rationale,
            "meta_learner_status": "LEARNED GATING: NOT TRAINED",
            "learned_gating_status": "LEARNED GATING: NOT TRAINED"
        }
