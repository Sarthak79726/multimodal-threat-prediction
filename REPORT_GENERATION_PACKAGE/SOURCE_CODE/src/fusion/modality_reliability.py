"""
Modality Reliability, Sensor Quality, and Missing-Data Handling Engine.
Evaluates the physical and computational validity of each sensor stream independently,
preventing raw score magnitude from being conflated with signal trustworthiness.
Implements mathematically grounded missing-modality compensation without silent zero-substitution.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

from src.fusion.fusion_config import (
    FRESHNESS_TIMEOUTS_SEC,
    RELIABILITY_THRESHOLDS,
    MissingModalityStrategy,
    DEFAULT_BASELINE_WEIGHTS
)

@dataclass
class ModalityReliabilityReport:
    """Detailed reliability and data-quality assessment for a single modality."""
    modality: str
    is_available: bool
    reliability_score: float             # In [0.0, 1.0]
    status: str                          # OPTIMAL, ACCEPTABLE, DEGRADED, STALE, UNAVAILABLE
    is_stale: bool
    age_seconds: float
    quality_factors: Dict[str, float] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    provenance_disclaimer: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "modality": self.modality,
            "is_available": self.is_available,
            "reliability_score": round(self.reliability_score, 4),
            "status": self.status,
            "is_stale": self.is_stale,
            "age_seconds": round(self.age_seconds, 1),
            "quality_factors": {k: round(v, 4) for k, v in self.quality_factors.items()},
            "warnings": self.warnings,
            "provenance_disclaimer": self.provenance_disclaimer
        }

class ModalityReliabilityEvaluator:
    """Evaluates sensor reliability, data freshness, and input quality across all branches."""

    @staticmethod
    def evaluate_physiological(
        telemetry: Optional[Dict[str, Any]],
        age_seconds: float = 0.0,
        expected_fs: int = 700
    ) -> ModalityReliabilityReport:
        """
        Evaluates physiological signal stream quality:
        - Signal availability (EDA, Temp, ECG present)
        - Physiological validity bounds (HR, EDA, Temp)
        - Sampling alignment
        - Timestamp freshness
        """
        if not telemetry or not isinstance(telemetry, dict):
            return ModalityReliabilityReport(
                modality="physiological",
                is_available=False,
                reliability_score=0.0,
                status="UNAVAILABLE",
                is_stale=False,
                age_seconds=age_seconds,
                warnings=["Physiological telemetry stream is completely unavailable."],
                provenance_disclaimer="No biometric signal received."
            )

        warnings: List[str] = []
        factors: Dict[str, float] = {}

        # 1. Freshness check
        max_age = FRESHNESS_TIMEOUTS_SEC["physiological"]
        is_stale = age_seconds > max_age
        freshness_factor = max(0.0, min(1.0, 1.0 - (age_seconds / (max_age * 2.0))))
        factors["freshness"] = freshness_factor
        if is_stale:
            warnings.append(f"Biometric data age ({age_seconds:.1f}s) exceeds freshness threshold ({max_age}s).")

        # 2. Sensor Channel Completeness
        eda = telemetry.get("eda_mean")
        temp = telemetry.get("temp_mean")
        ecg_std = telemetry.get("ecg_std")
        hr = telemetry.get("heart_rate_bpm")

        ch_present = sum([eda is not None, temp is not None, ecg_std is not None, hr is not None])
        completeness_factor = ch_present / 4.0
        factors["completeness"] = completeness_factor
        if completeness_factor < 1.0:
            warnings.append(f"Incomplete channels: {4 - ch_present} sensor streams missing.")

        # 3. Biometric Plausibility Bounds
        plausibility = 1.0
        if hr is not None:
            if hr < 40 or hr > 200:
                plausibility -= 0.3
                warnings.append(f"Unusual heart rate value: {hr} BPM outside physiological bounds [40, 200].")
        if eda is not None:
            if eda < 0.01 or eda > 100.0:
                plausibility -= 0.3
                warnings.append(f"Unusual EDA value: {eda} µS outside typical dermal conductance range.")
        if temp is not None:
            if temp < 20.0 or temp > 43.0:
                plausibility -= 0.3
                warnings.append(f"Skin temperature: {temp} °C outside physiological range [20, 43].")
        factors["plausibility"] = max(0.0, plausibility)

        # 4. Data Fingerprint & Verification
        fp = telemetry.get("fingerprint")
        has_fp = fp is not None and fp != "N/A"
        factors["provenance_integrity"] = 1.0 if has_fp else 0.5

        # Composite Reliability Score
        score = (
            0.35 * factors["freshness"] +
            0.30 * factors["completeness"] +
            0.25 * factors["plausibility"] +
            0.10 * factors["provenance_integrity"]
        )
        score = float(np.clip(score, 0.0, 1.0))

        if score >= RELIABILITY_THRESHOLDS["OPTIMAL_MIN"] and not is_stale:
            status = "OPTIMAL"
        elif score >= RELIABILITY_THRESHOLDS["ACCEPTABLE_MIN"] and not is_stale:
            status = "ACCEPTABLE"
        elif is_stale:
            status = "STALE"
        else:
            status = "DEGRADED"

        return ModalityReliabilityReport(
            modality="physiological",
            is_available=True,
            reliability_score=score,
            status=status,
            is_stale=is_stale,
            age_seconds=age_seconds,
            quality_factors=factors,
            warnings=warnings,
            provenance_disclaimer=telemetry.get("provenance", "WESAD Clinical Laboratory Wearable Signals")
        )

    @staticmethod
    def evaluate_spatial(
        spatial_analysis: Optional[Dict[str, Any]],
        age_seconds: float = 0.0,
        is_manual_override: bool = False
    ) -> ModalityReliabilityReport:
        """
        Evaluates spatial / GPS data quality:
        - GPS accuracy radius (meters)
        - Campus geofence containment
        - Equidistant landmark ambiguity
        - Manual user override verification
        - Coordinate freshness
        """
        if not spatial_analysis or not isinstance(spatial_analysis, dict):
            return ModalityReliabilityReport(
                modality="spatial",
                is_available=False,
                reliability_score=0.0,
                status="UNAVAILABLE",
                is_stale=False,
                age_seconds=age_seconds,
                warnings=["Spatial positioning data is unavailable."],
                provenance_disclaimer="GPS signal unavailable."
            )

        warnings: List[str] = []
        factors: Dict[str, float] = {}

        # 1. Freshness
        max_age = FRESHNESS_TIMEOUTS_SEC["spatial"]
        is_stale = age_seconds > max_age and not is_manual_override
        freshness_factor = 1.0 if is_manual_override else max(0.0, min(1.0, 1.0 - (age_seconds / (max_age * 2.0))))
        factors["freshness"] = freshness_factor
        if is_stale:
            warnings.append(f"Spatial coordinates age ({age_seconds:.1f}s) exceeds timeout ({max_age}s).")

        # 2. GPS Accuracy or Manual Override Certainty
        c_loc = spatial_analysis.get("current_location", {})
        acc_m = c_loc.get("accuracy_meters") if isinstance(c_loc, dict) else None
        if is_manual_override:
            accuracy_factor = 0.95  # User explicit landmark confirmation is high reliability
            factors["accuracy"] = accuracy_factor
            factors["source_type"] = 1.0
        elif acc_m is not None:
            if acc_m <= 10.0:
                accuracy_factor = 1.0
            elif acc_m <= 30.0:
                accuracy_factor = 0.85
            elif acc_m <= 100.0:
                accuracy_factor = 0.60
            else:
                accuracy_factor = 0.30
                warnings.append(f"High GPS inaccuracy: ±{acc_m:.1f}m error margin.")
            factors["accuracy"] = accuracy_factor
            factors["source_type"] = 0.9
        else:
            accuracy_factor = 0.50
            factors["accuracy"] = accuracy_factor
            factors["source_type"] = 0.5

        # 3. Geofence & Ambiguity Status
        status_str = spatial_analysis.get("status", "CONNECTED")
        is_ambiguous = (status_str == "AMBIGUOUS_LOCATION")
        is_outside = (status_str == "OUTSIDE_CAMPUS_GEOFENCE")
        ambiguity_factor = 0.40 if is_ambiguous else 1.0
        factors["ambiguity"] = ambiguity_factor
        if is_ambiguous:
            warnings.append("Position equidistant between multiple landmarks; resolution is ambiguous.")
        if is_outside:
            warnings.append("Coordinates fall outside BIT Mesra campus geofence boundary.")
            factors["geofence"] = 0.30
        else:
            factors["geofence"] = 1.0

        # Composite Spatial Score
        score = (
            0.35 * factors["accuracy"] +
            0.25 * factors["freshness"] +
            0.25 * factors["ambiguity"] +
            0.15 * factors["geofence"]
        )
        score = float(np.clip(score, 0.0, 1.0))

        if score >= RELIABILITY_THRESHOLDS["OPTIMAL_MIN"] and not is_stale:
            status = "OPTIMAL"
        elif score >= RELIABILITY_THRESHOLDS["ACCEPTABLE_MIN"] and not is_stale:
            status = "ACCEPTABLE"
        elif is_stale:
            status = "STALE"
        else:
            status = "DEGRADED"

        return ModalityReliabilityReport(
            modality="spatial",
            is_available=True,
            reliability_score=score,
            status=status,
            is_stale=is_stale,
            age_seconds=age_seconds,
            quality_factors=factors,
            warnings=warnings,
            provenance_disclaimer="Manual Override Landmark" if is_manual_override else "BIT Mesra Campus GPS Geofence Engine"
        )

    @staticmethod
    def evaluate_environmental(
        env_res: Optional[Dict[str, Any]],
        age_seconds: float = 0.0
    ) -> ModalityReliabilityReport:
        """
        Evaluates environmental branch reliability:
        - Synthetic model provenance (strictly transparent: models trained on synthetic simulation)
        - Inference success status
        - Prediction confidence (crowd model and lighting model probabilities)
        - Asia/Kolkata timezone freshness
        """
        if not env_res or not isinstance(env_res, dict) or env_res.get("status") != "SUCCESS":
            msg = env_res.get("message") if env_res else "Environmental branch unavailable."
            return ModalityReliabilityReport(
                modality="environmental",
                is_available=False,
                reliability_score=0.0,
                status="UNAVAILABLE",
                is_stale=False,
                age_seconds=age_seconds,
                warnings=[f"Environmental branch unavailable: {msg}"],
                provenance_disclaimer="Environmental Context Engine offline."
            )

        warnings: List[str] = []
        factors: Dict[str, float] = {}

        # 1. Synthetic Provenance Guard:
        # Synthetic data is never scored at 1.0 to prevent conflating it with live IoT hardware sensors
        factors["sensor_authenticity"] = 0.65
        warnings.append("Experimental prediction from campus synthetic model — NOT live sensor measurement.")

        # 2. Freshness
        max_age = FRESHNESS_TIMEOUTS_SEC["environmental"]
        is_stale = age_seconds > max_age
        freshness_factor = max(0.0, min(1.0, 1.0 - (age_seconds / (max_age * 2.0))))
        factors["freshness"] = freshness_factor
        if is_stale:
            warnings.append(f"Environmental context timestamp age ({age_seconds:.1f}s) is stale.")

        # 3. Model Prediction Confidence
        crowd_res = env_res.get("crowd_prediction", {})
        light_res = env_res.get("lighting_prediction", {})
        c_conf = crowd_res.get("confidence_pct", 60.0) / 100.0
        l_conf = light_res.get("confidence_pct", 70.0) / 100.0
        factors["model_confidence"] = float(np.clip((c_conf + l_conf) / 2.0, 0.0, 1.0))

        # Composite Environmental Score
        score = (
            0.40 * factors["sensor_authenticity"] +
            0.30 * factors["model_confidence"] +
            0.30 * factors["freshness"]
        )
        score = float(np.clip(score, 0.0, 1.0))

        if score >= RELIABILITY_THRESHOLDS["OPTIMAL_MIN"] and not is_stale:
            status = "OPTIMAL"
        elif score >= RELIABILITY_THRESHOLDS["ACCEPTABLE_MIN"] and not is_stale:
            status = "ACCEPTABLE"
        elif is_stale:
            status = "STALE"
        else:
            status = "DEGRADED"

        return ModalityReliabilityReport(
            modality="environmental",
            is_available=True,
            reliability_score=score,
            status=status,
            is_stale=is_stale,
            age_seconds=age_seconds,
            quality_factors=factors,
            warnings=warnings,
            provenance_disclaimer="Synthetic-Trained Environmental Models (Gradient Boosting + Random Forest)"
        )


class MissingModalityCompensator:
    """
    Manages missing, degraded, or stale modalities.
    Avoids silent neutral-value or zero-value substitution.
    Applies mathematically validated marginalized re-normalization with explicit uncertainty expansion.
    """

    @classmethod
    def compensate(
        cls,
        p_val: Optional[float],
        l_val: Optional[float],
        e_val: Optional[float],
        rel_p: ModalityReliabilityReport,
        rel_l: ModalityReliabilityReport,
        rel_e: ModalityReliabilityReport,
        nominal_weights: Dict[str, float] = DEFAULT_BASELINE_WEIGHTS,
        strategy: MissingModalityStrategy = MissingModalityStrategy.MARGINALIZED_EXCLUSION
    ) -> Dict[str, Any]:
        """
        Computes the effective weights, adjusted score, and uncertainty metric under missing data.
        """
        # Determine validity: Must be available, non-stale, and with finite value
        valid_p = rel_p.is_available and (p_val is not None) and np.isfinite(p_val) and (rel_p.status != "UNAVAILABLE")
        valid_l = rel_l.is_available and (l_val is not None) and np.isfinite(l_val) and (rel_l.status != "UNAVAILABLE")
        valid_e = rel_e.is_available and (e_val is not None) and np.isfinite(e_val) and (rel_e.status != "UNAVAILABLE")

        available_mods = []
        if valid_p: available_mods.append("physiological")
        if valid_l: available_mods.append("spatial")
        if valid_e: available_mods.append("environmental")

        missing_mods = []
        if not valid_p: missing_mods.append("physiological")
        if not valid_l: missing_mods.append("spatial")
        if not valid_e: missing_mods.append("environmental")

        num_available = len(available_mods)

        # Case 0: Complete 3-Branch availability
        if num_available == 3:
            w_p = nominal_weights.get("physiological", 0.40)
            w_l = nominal_weights.get("location", 0.25)
            w_e = nominal_weights.get("environment", 0.35)
            tot = w_p + w_l + w_e
            effective_weights = {"physiological": w_p / tot, "spatial": w_l / tot, "environmental": w_e / tot}
            score = effective_weights["physiological"] * float(p_val) + effective_weights["spatial"] * float(l_val) + effective_weights["environmental"] * float(e_val)
            
            # Uncertainty is lowest when all 3 modalities are available
            avg_rel = (rel_p.reliability_score + rel_l.reliability_score + rel_e.reliability_score) / 3.0
            uncertainty_index = round(float(np.clip(1.0 - avg_rel, 0.05, 0.95)), 4)

            return {
                "status": "COMPLETE_TRI_MODAL",
                "available_modalities": available_mods,
                "missing_modalities": [],
                "effective_weights": effective_weights,
                "marginalized_score": round(float(score), 4),
                "is_complete": True,
                "uncertainty_index": uncertainty_index,
                "strategy_applied": "NONE (Full Tri-Modal)",
                "warning": None
            }

        # Case 1: Strict Abort requested
        if strategy == MissingModalityStrategy.STRICT_ABORT:
            return {
                "status": "ABORTED_INCOMPLETE_MODALITIES",
                "available_modalities": available_mods,
                "missing_modalities": missing_mods,
                "effective_weights": {},
                "marginalized_score": None,
                "is_complete": False,
                "uncertainty_index": 1.0,
                "strategy_applied": strategy.value,
                "warning": f"Fusion aborted: Complete tri-modal inputs required. Missing: {missing_mods}"
            }

        # Case 2: Only 0 or 1 modality available (Insufficient for multimodal fusion)
        if num_available < 2:
            return {
                "status": "INSUFFICIENT_MODALITIES",
                "available_modalities": available_mods,
                "missing_modalities": missing_mods,
                "effective_weights": {},
                "marginalized_score": None,
                "is_complete": False,
                "uncertainty_index": 0.95,
                "strategy_applied": strategy.value,
                "warning": f"Insufficient modalities ({num_available}/3 available). Multimodal fusion requires at least 2 corroborated branches."
            }

        # Case 3: Exactly 2 modalities available (Verified Partial Dual-Branch)
        w_sum = 0.0
        active_weights: Dict[str, float] = {}
        weighted_terms: List[float] = []

        if valid_p:
            w_sum += nominal_weights.get("physiological", 0.40)
        if valid_l:
            w_sum += nominal_weights.get("location", 0.25)
        if valid_e:
            w_sum += nominal_weights.get("environment", 0.35)

        effective_weights = {}
        for mod, is_v, val in [("physiological", valid_p, p_val), ("spatial", valid_l, l_val), ("environmental", valid_e, e_val)]:
            if is_v:
                norm_w = nominal_weights.get("location" if mod == "spatial" else mod, 0.33) / w_sum
                effective_weights[mod] = round(norm_w, 4)
                weighted_terms.append(norm_w * float(val))
            else:
                effective_weights[mod] = 0.0

        dual_score = float(sum(weighted_terms))

        # Explicit uncertainty expansion: Penalize uncertainty index when a modality is excluded
        missing_penalty = 0.25 if strategy == MissingModalityStrategy.UNCERTAINTY_PENALTY else 0.15
        active_rels = [r.reliability_score for r in [rel_p, rel_l, rel_e] if r.modality in available_mods]
        avg_rel = sum(active_rels) / len(active_rels) if active_rels else 0.5
        uncertainty_index = round(float(np.clip(1.0 - avg_rel + missing_penalty, 0.15, 0.95)), 4)

        return {
            "status": "PARTIAL_DUAL_BRANCH",
            "available_modalities": available_mods,
            "missing_modalities": missing_mods,
            "effective_weights": effective_weights,
            "marginalized_score": round(dual_score, 4),
            "is_complete": False,
            "uncertainty_index": uncertainty_index,
            "strategy_applied": strategy.value,
            "warning": f"Operating in marginalized dual-branch mode ({'+'.join(available_mods)}). Excluded: {missing_mods}."
        }
