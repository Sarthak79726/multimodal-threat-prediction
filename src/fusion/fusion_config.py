"""
Configuration, Enumerations, and Provenance Metadata for Dynamic Multimodal Fusion.
Defines fusion modes, missing modality handling strategies, candidate risk categories,
quality thresholds, and institutional campus safety policies.
"""

from enum import Enum
from typing import Dict, Any, List

class FusionMode(str, Enum):
    """Supported multimodal fusion paradigms."""
    STATIC_WEIGHTED = "STATIC_WEIGHTED"       # Operational Baseline: Fixed weights (0.40P + 0.25L + 0.35E)
    DYNAMIC_GATING = "DYNAMIC_GATING"         # Learned Dynamic Gating Network: alpha = Softmax(g(X))
    META_LEARNER = "META_LEARNER"             # Level 1 Stacking Meta-Learner (Logistic Regression / XGBoost)

class MissingModalityStrategy(str, Enum):
    """Explicit, mathematically sound strategies for handling missing or corrupted modalities."""
    MARGINALIZED_EXCLUSION = "MARGINALIZED_EXCLUSION"  # Re-normalize remaining valid modalities with uncertainty expansion
    UNCERTAINTY_PENALTY = "UNCERTAINTY_PENALTY"        # Apply explicit uncertainty penalty to overall assessment
    STRICT_ABORT = "STRICT_ABORT"                      # Reject assessment if complete tri-modal inputs are not available

class CandidateRiskCategory(str, Enum):
    """
    Independently defined candidate risk categories for multi-output prediction.
    These represent separate operational targets rather than forcing all phenomena into
    one generic monolithic threat category.
    """
    PHYSIOLOGICAL_DISTRESS = "PHYSIOLOGICAL_DISTRESS"    # Wearable biometrics: acute physiological arousal/distress
    CROWD_CONGESTION = "CROWD_CONGESTION"                # Environmental: dense crowd accumulation / bottleneck
    ENVIRONMENTAL_HAZARD = "ENVIRONMENTAL_HAZARD"        # Situational: low illumination / high situational vulnerability
    LOCATION_EXPOSURE = "LOCATION_EXPOSURE"              # Spatial: isolated zone / distance to nearest security post
    VERIFIED_CAMPUS_INCIDENT = "VERIFIED_CAMPUS_INCIDENT" # Ground truth incident (requires institutional reporting)

# Baseline static fusion weights (Reference Benchmark)
DEFAULT_BASELINE_WEIGHTS: Dict[str, float] = {
    "physiological": 0.40,
    "location": 0.25,
    "environment": 0.35
}

# Context categorization thresholds
CONTEXT_THRESHOLDS: Dict[str, float] = {
    "LOW_MAX": 0.39,      # Standard Context
    "MEDIUM_MAX": 0.69,   # Elevated Context
}                         # > 0.69: High Context

# Modality freshness and staleness timeouts (in seconds)
FRESHNESS_TIMEOUTS_SEC: Dict[str, float] = {
    "physiological": 120.0,   # Biometric signals older than 2 minutes are stale
    "spatial": 300.0,         # GPS coordinates older than 5 minutes are stale
    "environmental": 900.0    # Environmental situational context older than 15 minutes is stale
}

# Reliability score thresholds (0.0 to 1.0)
RELIABILITY_THRESHOLDS: Dict[str, float] = {
    "OPTIMAL_MIN": 0.80,
    "ACCEPTABLE_MIN": 0.50,
    "DEGRADED_MIN": 0.30
}

# Modality scientific provenance definitions
MODALITY_PROVENANCE: Dict[str, Dict[str, Any]] = {
    "physiological": {
        "modality": "Physiological Stress",
        "dataset": "WESAD Clinical Laboratory Benchmark",
        "sensor_type": "RespiBAN Chest (700Hz) & Empatica E4 Wrist",
        "semantic_scope": "Affective Stress & Emotional Arousal (NOT real-world physical threat)",
        "is_synthetic": False,
        "is_subject_isolated": True
    },
    "spatial": {
        "modality": "Spatial Context & Location",
        "dataset": "BIT Mesra Campus Vector Geospatial Database",
        "sensor_type": "Browser GPS / Geolocation API & Coordinate Presets",
        "semantic_scope": "Relative Distance to Security Posts & Campus Geofence (NOT crime incidence)",
        "is_synthetic": False,
        "is_subject_isolated": False
    },
    "environmental": {
        "modality": "Environmental Situational Context",
        "dataset": "BIT Mesra Campus Synthetic Environmental Simulation",
        "sensor_type": "Trained XGBoost & Random Forest Models (Inference on Time/Location)",
        "semantic_scope": "Estimated Crowd Density & Illumination Lux (NOT live hardware sensor measurement)",
        "is_synthetic": True,
        "is_subject_isolated": False
    }
}

# Institutional safety, privacy and ethical review notice
CAMPUS_SAFETY_POLICY: Dict[str, Any] = {
    "intended_use": "Situational campus welfare awareness and human-in-the-loop safety coordination.",
    "explicit_prohibition": "Automated punitive actions, accusations, or automated classification of individuals as dangerous.",
    "privacy_principles": [
        "Explicit user consent required for biometric signal streaming.",
        "Data minimization: Raw waveforms stored locally; only aggregated features transmitted.",
        "Zero long-term retention of raw physiological signals without explicit research approval.",
        "User opt-out supported at all times via dashboard controls."
    ],
    "institutional_review_status": "Academic Research & Educational Prototype — Clinical/Police deployment requires IRB approval."
}
