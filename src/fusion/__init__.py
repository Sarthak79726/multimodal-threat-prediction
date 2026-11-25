"""
Multimodal Threat Prediction — Fusion Subsystem Package.
Provides static weighted baseline fusion, dynamic gating networks, stacking meta-learners,
reliability evaluators, probability calibration, and multi-output risk assessment.
"""

from src.fusion.fusion_config import (
    FusionMode,
    MissingModalityStrategy,
    CandidateRiskCategory,
    DEFAULT_BASELINE_WEIGHTS,
    CONTEXT_THRESHOLDS,
    MODALITY_PROVENANCE,
    CAMPUS_SAFETY_POLICY
)
from src.fusion.feature_schema import (
    GATING_FEATURE_NAMES,
    META_FEATURE_NAMES,
    build_gating_feature_vector,
    validate_gating_vector
)
from src.fusion.modality_reliability import (
    ModalityReliabilityReport,
    ModalityReliabilityEvaluator,
    MissingModalityCompensator
)
from src.fusion.dynamic_gating import DynamicGatingNetwork
from src.fusion.meta_learner import (
    BaseMetaLearner,
    LogisticRegressionMetaLearner,
    RandomForestMetaLearner,
    XGBoostMetaLearner,
    WeightedLateFusionBaseline,
    MultiOutputMetaLearner
)
from src.fusion.calibration import ProbabilityCalibrator
from src.fusion.fusion_engine import DynamicMultimodalFusionEngine
from src.fusion.adaptive_fusion import (
    ReliabilityAwareAdaptiveFusionEngine,
    ModalityReliabilityMetrics,
    BASELINE_PRIORS,
    FRESHNESS_TAUS_SEC
)

__all__ = [
    "FusionMode",
    "MissingModalityStrategy",
    "CandidateRiskCategory",
    "DEFAULT_BASELINE_WEIGHTS",
    "CONTEXT_THRESHOLDS",
    "MODALITY_PROVENANCE",
    "CAMPUS_SAFETY_POLICY",
    "GATING_FEATURE_NAMES",
    "META_FEATURE_NAMES",
    "build_gating_feature_vector",
    "validate_gating_vector",
    "ModalityReliabilityReport",
    "ModalityReliabilityEvaluator",
    "MissingModalityCompensator",
    "DynamicGatingNetwork",
    "BaseMetaLearner",
    "LogisticRegressionMetaLearner",
    "RandomForestMetaLearner",
    "XGBoostMetaLearner",
    "WeightedLateFusionBaseline",
    "MultiOutputMetaLearner",
    "ProbabilityCalibrator",
    "DynamicMultimodalFusionEngine",
    "ReliabilityAwareAdaptiveFusionEngine",
    "ModalityReliabilityMetrics",
    "BASELINE_PRIORS",
    "FRESHNESS_TAUS_SEC"
]
