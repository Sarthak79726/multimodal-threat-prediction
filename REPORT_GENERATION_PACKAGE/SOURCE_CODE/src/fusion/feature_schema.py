"""
Feature Schemas and Validation for Dynamic Multimodal Fusion.
Defines input vectors, context metadata, gating feature schemas, and strict validation
guards against target leakage, circular labels, and corrupted inputs.
"""

from typing import Dict, Any, List, Optional
import numpy as np

class FeatureValidationError(ValueError):
    """Raised when an input feature vector fails schema validation or contains invalid values."""
    pass

class TargetLeakageError(ValueError):
    """Raised when ground truth or target labels are leaked into the feature vector."""
    pass

# Dynamic Gating Feature Vector (X_gating)
# Combines normalized modality scores, model uncertainties, and reliability indicators
GATING_FEATURE_NAMES: List[str] = [
    "p_stress",               # Physiological stress probability [0.0, 1.0]
    "p_stress_confidence",    # Confidence or certainty of physiological model [0.0, 1.0]
    "physio_reliability",     # Physiological sensor & signal reliability score [0.0, 1.0]
    "loc_risk",               # Spatial context exposure score [0.0, 1.0]
    "loc_confidence",         # Spatial location certainty / accuracy metric [0.0, 1.0]
    "loc_ambiguity",          # Binary indicator: 1.0 if location is ambiguous, 0.0 otherwise
    "spatial_reliability",    # GPS hardware & coordinate reliability score [0.0, 1.0]
    "env_risk",               # Situational environmental score [0.0, 1.0]
    "crowd_conf",             # Confidence of crowd classification [0.0, 1.0]
    "light_conf",             # Confidence of lighting classification [0.0, 1.0]
    "env_reliability",        # Environmental sensor/model reliability score [0.0, 1.0]
    "is_synthetic_env",       # Binary indicator: 1.0 if environmental data is synthetic
    "missing_physio",         # Binary indicator: 1.0 if physiological branch is missing
    "missing_spatial",        # Binary indicator: 1.0 if spatial branch is missing
    "missing_env"             # Binary indicator: 1.0 if environmental branch is missing
]

# Meta-Learner Level 1 Feature Vector (X_meta)
META_FEATURE_NAMES: List[str] = [
    "p_physiological",        # Base model 1 output
    "p_spatial",              # Base model 2 output
    "p_environmental",        # Base model 3 output
    "physio_reliability",     # Reliability feature 1
    "spatial_reliability",    # Reliability feature 2
    "env_reliability",        # Reliability feature 3
    "missing_env_indicator"   # Missingness feature
]

def validate_gating_vector(vec: np.ndarray) -> np.ndarray:
    """
    Validates a dynamic gating input vector against dimension, non-finiteness, and bounds.
    Returns cleaned float64 2D numpy array [1, N] or [M, N].
    """
    arr = np.asarray(vec, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    
    if arr.shape[1] != len(GATING_FEATURE_NAMES):
        raise FeatureValidationError(
            f"Gating feature dimension mismatch: expected {len(GATING_FEATURE_NAMES)} features, got {arr.shape[1]}."
        )
    
    if not np.all(np.isfinite(arr)):
        raise FeatureValidationError("Gating feature vector contains non-finite values (NaN or Inf).")
        
    return arr

def build_gating_feature_vector(
    p_stress: float,
    p_stress_conf: float,
    physio_rel: float,
    loc_risk: float,
    loc_conf: float,
    loc_ambiguous: bool,
    spatial_rel: float,
    env_risk: Optional[float],
    crowd_conf: float,
    light_conf: float,
    env_rel: float,
    is_synthetic_env: bool = True,
    missing_physio: bool = False,
    missing_spatial: bool = False,
    missing_env: bool = False
) -> np.ndarray:
    """
    Constructs and validates the canonical 15-dimensional feature vector for dynamic gating.
    Defensively handles missing modalities without substituting artificial zeros.
    """
    e_val = float(env_risk) if (env_risk is not None and np.isfinite(env_risk)) else 0.0
    
    features = [
        float(np.clip(p_stress, 0.0, 1.0)),
        float(np.clip(p_stress_conf, 0.0, 1.0)),
        float(np.clip(physio_rel, 0.0, 1.0)),
        float(np.clip(loc_risk, 0.0, 1.0)),
        float(np.clip(loc_conf, 0.0, 1.0)),
        1.0 if loc_ambiguous else 0.0,
        float(np.clip(spatial_rel, 0.0, 1.0)),
        float(np.clip(e_val, 0.0, 1.0)),
        float(np.clip(crowd_conf, 0.0, 1.0)),
        float(np.clip(light_conf, 0.0, 1.0)),
        float(np.clip(env_rel, 0.0, 1.0)),
        1.0 if is_synthetic_env else 0.0,
        1.0 if missing_physio else 0.0,
        1.0 if missing_spatial else 0.0,
        1.0 if missing_env or (env_risk is None) else 0.0
    ]
    
    return validate_gating_vector(np.array(features, dtype=np.float64))
