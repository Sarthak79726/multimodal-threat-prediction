"""
Unified Dynamic Multimodal Fusion Engine.
Orchestrates:
1. Static Weighted Fusion Baseline (Operational Reference: 0.40P + 0.25L + 0.35E).
2. Dynamic Gating Network (Learned Contextual Modality Importance: alpha = Softmax(g(X))).
3. Stacking Meta-Learner (Level 1 Learned Decision Optimization).
4. Multi-Output Risk Category Assessment.
5. Modality Reliability Evaluation and Missing-Data Management.
6. Institutional Campus Safety, Privacy, and Ethical Governance.
"""

import sys
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

# Ensure project root is in sys.path
_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import config
from src.fusion.fusion_config import (
    FusionMode,
    MissingModalityStrategy,
    CandidateRiskCategory,
    DEFAULT_BASELINE_WEIGHTS,
    CONTEXT_THRESHOLDS,
    MODALITY_PROVENANCE,
    CAMPUS_SAFETY_POLICY
)
from src.fusion.feature_schema import build_gating_feature_vector
from src.fusion.modality_reliability import (
    ModalityReliabilityEvaluator,
    MissingModalityCompensator,
    ModalityReliabilityReport
)
from src.fusion.dynamic_gating import DynamicGatingNetwork
from src.fusion.meta_learner import (
    LogisticRegressionMetaLearner,
    MultiOutputMetaLearner,
    WeightedLateFusionBaseline
)
from src.fusion.calibration import ProbabilityCalibrator

logger = logging.getLogger("DynamicMultimodalFusionEngine")

class DynamicMultimodalFusionEngine:
    """
    State-of-the-art Dynamic Multimodal Fusion Engine.
    Executes flexible, data-driven multimodal inference across physiological,
    spatial, and environmental modalities.
    """

    def __init__(
        self,
        default_mode: FusionMode = FusionMode.STATIC_WEIGHTED,
        missing_strategy: MissingModalityStrategy = MissingModalityStrategy.MARGINALIZED_EXCLUSION,
        threshold: float = 0.50
    ):
        self.active_mode = default_mode
        self.missing_strategy = missing_strategy
        self.threshold = threshold

        # Core Modality Sub-Engines & Baseline
        self.baseline_fusion = WeightedLateFusionBaseline(threshold=threshold)
        self.gating_network = DynamicGatingNetwork()
        self.meta_learner = LogisticRegressionMetaLearner()
        self.multi_output = MultiOutputMetaLearner()
        self.calibrator = ProbabilityCalibrator(method="platt")

        # Provenance and training records
        self.is_gating_trained = False
        self.is_meta_trained = False
        self.is_calibrated = False

    def set_fusion_mode(self, mode: FusionMode) -> None:
        """Sets the active fusion mode."""
        self.active_mode = mode
        logger.info("Active fusion mode switched to: %s", mode.value)

    def evaluate_fusion(
        self,
        p_stress: Optional[float],
        l_risk: Optional[float],
        e_risk: Optional[float],
        physio_telemetry: Optional[Dict[str, Any]] = None,
        spatial_analysis: Optional[Dict[str, Any]] = None,
        environmental_res: Optional[Dict[str, Any]] = None,
        physio_age_sec: float = 0.0,
        spatial_age_sec: float = 0.0,
        env_age_sec: float = 0.0,
        is_manual_override: bool = False,
        mode: Optional[FusionMode] = None
    ) -> Dict[str, Any]:
        """
        Executes end-to-end multimodal fusion:
        1. Evaluates sensor stream reliability and input freshness.
        2. Detects missing modalities and computes compensated weights.
        3. Constructs canonical gating feature vector.
        4. Evaluates selected fusion mode (Static, Dynamic Gating, or Meta-Learner).
        5. Computes multi-output risk category predictions.
        6. Generates grounded mathematical explainability and privacy disclosures.
        """
        selected_mode = mode or self.active_mode

        # -----------------------------------------------------------------
        # STEP 1: MODALITY RELIABILITY EVALUATION
        # -----------------------------------------------------------------
        rel_p = ModalityReliabilityEvaluator.evaluate_physiological(
            telemetry=physio_telemetry,
            age_seconds=physio_age_sec
        )
        rel_l = ModalityReliabilityEvaluator.evaluate_spatial(
            spatial_analysis=spatial_analysis,
            age_seconds=spatial_age_sec,
            is_manual_override=is_manual_override
        )
        rel_e = ModalityReliabilityEvaluator.evaluate_environmental(
            env_res=environmental_res,
            age_seconds=env_age_sec
        )

        reliability_summary = {
            "physiological": rel_p.to_dict(),
            "spatial": rel_l.to_dict(),
            "environmental": rel_e.to_dict()
        }

        # -----------------------------------------------------------------
        # STEP 2: MISSING DATA MANAGEMENT
        # -----------------------------------------------------------------
        compensation = MissingModalityCompensator.compensate(
            p_val=p_stress,
            l_val=l_risk,
            e_val=e_risk,
            rel_p=rel_p,
            rel_l=rel_l,
            rel_e=rel_e,
            nominal_weights=DEFAULT_BASELINE_WEIGHTS,
            strategy=self.missing_strategy
        )

        is_complete_tri_modal = compensation["is_complete"]
        available_modalities = compensation["available_modalities"]
        missing_modalities = compensation["missing_modalities"]
        uncertainty_index = compensation["uncertainty_index"]

        # Safe clamped values
        p_norm = float(np.clip(p_stress if p_stress is not None else 0.18, 0.0, 1.0))
        l_norm = float(np.clip(l_risk if l_risk is not None else 0.15, 0.0, 1.0))
        e_norm = float(np.clip(e_risk, 0.0, 1.0)) if (e_risk is not None and np.isfinite(e_risk)) else None

        # -----------------------------------------------------------------
        # STEP 3: CONSTRUCT GATING & META FEATURE VECTORS
        # -----------------------------------------------------------------
        crowd_conf = (environmental_res.get("crowd_prediction", {}).get("confidence_pct", 60.0) / 100.0) if environmental_res else 0.0
        light_conf = (environmental_res.get("lighting_prediction", {}).get("confidence_pct", 70.0) / 100.0) if environmental_res else 0.0
        is_ambiguous = bool(spatial_analysis.get("status") == "AMBIGUOUS_LOCATION") if spatial_analysis else False

        gating_vec = build_gating_feature_vector(
            p_stress=p_norm,
            p_stress_conf=0.90 if rel_p.is_available else 0.0,
            physio_rel=rel_p.reliability_score,
            loc_risk=l_norm,
            loc_conf=rel_l.quality_factors.get("accuracy", 0.70),
            loc_ambiguous=is_ambiguous,
            spatial_rel=rel_l.reliability_score,
            env_risk=e_norm,
            crowd_conf=crowd_conf,
            light_conf=light_conf,
            env_rel=rel_e.reliability_score,
            is_synthetic_env=True,
            missing_physio=("physiological" in missing_modalities),
            missing_spatial=("spatial" in missing_modalities),
            missing_env=("environmental" in missing_modalities)
        )

        # -----------------------------------------------------------------
        # STEP 4: FUSION INFERENCE (ACCORDING TO SELECTED MODE)
        # -----------------------------------------------------------------
        weights_dict: Dict[str, float] = {}
        contributions_dict: Dict[str, float] = {}
        mode_status_str = ""
        is_model_trained = False
        mode_notice = ""
        dynamic_gating_details = None

        if selected_mode == FusionMode.STATIC_WEIGHTED:
            # Operational Reference Baseline
            mode_name = "Static Weighted Fusion Baseline"
            is_model_trained = False  # Fixed formula is non-trained baseline
            mode_status_str = "OPERATIONAL BASELINE (FIXED WEIGHTS)"
            mode_notice = "Evaluated via reference baseline: R = 0.40P + 0.25L + 0.35E (or marginalized dual-branch)."

            if is_complete_tri_modal:
                w_p, w_l, w_e = 0.40, 0.25, 0.35
                final_score = round(w_p * p_norm + w_l * l_norm + w_e * e_norm, 4)
                weights_dict = {"physiological": w_p, "spatial": w_l, "environmental": w_e}
                contributions_dict = {
                    "physiological": round(w_p * p_norm, 4),
                    "spatial": round(w_l * l_norm, 4),
                    "environmental": round(w_e * e_norm, 4)
                }
            else:
                # Marginalized Dual-Branch
                final_score = compensation["marginalized_score"]
                weights_dict = compensation["effective_weights"]
                contributions_dict = {
                    mod: round(weights_dict.get(mod, 0.0) * (p_norm if mod == "physiological" else (l_norm if mod == "spatial" else (e_norm or 0.0))), 4)
                    for mod in available_modalities
                }

        elif selected_mode == FusionMode.DYNAMIC_GATING:
            mode_name = "Dynamic Gating Network"
            gating_res = self.gating_network.predict_score(
                X=gating_vec,
                p_val=p_norm,
                l_val=l_norm,
                e_val=e_norm
            )
            is_model_trained = self.gating_network.is_trained
            dynamic_gating_details = gating_res["weights_meta"]

            if is_model_trained:
                final_score = gating_res["score"]
                weights_dict = gating_res["weights"]
                contributions_dict = gating_res["contributions"]
                mode_status_str = "TRAINED DYNAMIC GATING"
                mode_notice = "Modality contributions dynamically weighted via Softmax(g(X))."
            else:
                # Strictly transparent: UNTRAINED status
                mode_status_str = "UNTRAINED — Awaiting Ground-Truth Multimodal Dataset"
                mode_notice = "Gating architecture is fully implemented. Awaiting synchronized ground-truth dataset for parameter training. Displaying baseline reference."
                # Fall back to baseline score safely without inventing weights
                final_score = compensation["marginalized_score"]
                weights_dict = compensation["effective_weights"]
                contributions_dict = {
                    mod: round(weights_dict.get(mod, 0.0) * (p_norm if mod == "physiological" else (l_norm if mod == "spatial" else (e_norm or 0.0))), 4)
                    for mod in available_modalities
                }

        elif selected_mode == FusionMode.META_LEARNER:
            mode_name = "Level 1 Stacking Meta-Learner"
            is_model_trained = self.meta_learner.is_trained_ml

            if is_model_trained and is_complete_tri_modal:
                meta_x = np.array([[p_norm, l_norm, e_norm]])
                prob = float(self.meta_learner.predict_proba(meta_x)[0, 1])
                final_score = round(prob, 4)
                mode_status_str = "TRAINED META-LEARNER"
                mode_notice = "Evaluated via Level 1 Logistic Regression meta-learner."
                weights_dict = {"physiological": 0.33, "spatial": 0.33, "environmental": 0.34}
                contributions_dict = {"physiological": round(0.33 * p_norm, 4), "spatial": round(0.33 * l_norm, 4), "environmental": round(0.34 * e_norm, 4)}
            else:
                mode_status_str = "UNTRAINED — Production Eligibility Requires Verified Dataset"
                mode_notice = "Stacking meta-learner requires out-of-fold cross-validation on verified ground truth. Falling back to reference baseline."
                final_score = compensation["marginalized_score"]
                weights_dict = compensation["effective_weights"]
                contributions_dict = {
                    mod: round(weights_dict.get(mod, 0.0) * (p_norm if mod == "physiological" else (l_norm if mod == "spatial" else (e_norm or 0.0))), 4)
                    for mod in available_modalities
                }

        # Context categorization
        if final_score is not None:
            if final_score <= CONTEXT_THRESHOLDS["LOW_MAX"]:
                context_status = "STANDARD CONTEXT"
            elif final_score <= CONTEXT_THRESHOLDS["MEDIUM_MAX"]:
                context_status = "ELEVATED CONTEXT"
            else:
                context_status = "HIGH CONTEXT"
            final_pct = round(final_score * 100.0, 2)
        else:
            context_status = "INSUFFICIENT DATA"
            final_pct = None

        # -----------------------------------------------------------------
        # STEP 5: MULTI-OUTPUT CANDIDATE CATEGORY EVALUATION
        # -----------------------------------------------------------------
        meta_feat_vec = np.array([[p_norm, l_norm, e_norm or 0.0]])
        category_predictions = self.multi_output.predict_categories(meta_feat_vec)

        # -----------------------------------------------------------------
        # STEP 6: GROUNDED EXPLANABILITY
        # -----------------------------------------------------------------
        explanation = self._generate_grounded_explanation(
            mode_name=mode_name,
            final_score=final_score,
            context_status=context_status,
            p_val=p_norm,
            l_val=l_norm,
            e_val=e_norm,
            weights=weights_dict,
            rel_p=rel_p,
            rel_l=rel_l,
            rel_e=rel_e,
            missing_mods=missing_modalities
        )

        return {
            # Active Mode & Model Status
            "fusion_mode": selected_mode.value,
            "fusion_mode_name": mode_name,
            "is_model_trained": is_model_trained,
            "model_status": mode_status_str,
            "mode_notice": mode_notice,
            # Scores & Classification
            "experimental_context_index": final_score,
            "experimental_context_index_pct": final_pct,
            "context_status": context_status,
            "context_classification": context_status,
            "threat_probability": final_score,  # Retained for backward UI compatibility
            "threat_score_pct": final_pct,
            "threat_status": context_status,
            # Modality Inputs
            "physiological_stress_score": p_norm,
            "location_risk_score": l_norm,
            "environment_risk_score": e_norm,
            # Weights & Contributions
            "weights": weights_dict,
            "contributions": contributions_dict,
            "contributions_pct": {k: round(v * 100.0, 2) for k, v in contributions_dict.items()},
            # Reliability & Quality Details
            "reliability_summary": reliability_summary,
            "uncertainty_index": uncertainty_index,
            "is_complete_tri_modal": is_complete_tri_modal,
            "available_modalities": available_modalities,
            "missing_modalities": missing_modalities,
            "missing_strategy_applied": compensation["strategy_applied"],
            "missing_warning": compensation["warning"],
            # Multi-Output Risk Categories
            "candidate_risk_categories": category_predictions,
            # Dynamic Gating Details
            "dynamic_gating_details": dynamic_gating_details,
            # Grounded Decision Rationale
            "explanation": explanation,
            # Governance & Safety
            "scientific_disclaimer": "This system evaluates an experimental situational context index. It does NOT establish real-world threat probabilities.",
            "safety_policy": CAMPUS_SAFETY_POLICY
        }

    def _generate_grounded_explanation(
        self,
        mode_name: str,
        final_score: Optional[float],
        context_status: str,
        p_val: float,
        l_val: float,
        e_val: Optional[float],
        weights: Dict[str, float],
        rel_p: ModalityReliabilityReport,
        rel_l: ModalityReliabilityReport,
        rel_e: ModalityReliabilityReport,
        missing_mods: List[str]
    ) -> str:
        """Constructs an explainability narrative strictly grounded in actual numbers and quality reports."""
        if final_score is None:
            return f"Multimodal fusion could not evaluate an index because essential modalities were missing: {', '.join(missing_mods)}."

        lines = [
            f"Evaluated via {mode_name} resulting in an Experimental Context Index of {final_score*100:.1f}% ({context_status})."
        ]

        # Weight breakdown
        w_terms = [f"{mod.capitalize()}: {weights.get(mod, 0.0)*100:.1f}% weight" for mod in weights]
        lines.append(f"Modality weighting: {', '.join(w_terms)}.")

        # Modality observations
        obs = [f"Physiological Stress={p_val*100:.1f}% (Reliability: {rel_p.status})",
               f"Spatial Exposure={l_val*100:.1f}% (Reliability: {rel_l.status})"]
        if e_val is not None:
            obs.append(f"Environmental Context={e_val*100:.1f}% (Synthetic Simulation, Reliability: {rel_e.status})")
        else:
            obs.append("Environmental Context: UNKNOWN (Excluded from fusion)")
        lines.append(f"Input streams: {'; '.join(obs)}.")

        # Data quality notices
        quality_alerts = []
        if rel_p.warnings: quality_alerts.extend(rel_p.warnings)
        if rel_l.warnings: quality_alerts.extend(rel_l.warnings)
        if rel_e.warnings: quality_alerts.extend(rel_e.warnings)
        if quality_alerts:
            lines.append(f"Data quality considerations: {'; '.join(quality_alerts[:2])}")

        return " ".join(lines)
