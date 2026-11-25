"""
Physiological Stress Model Module.
Wraps the WESAD clinical biometrics stress classifier with a standardized prediction interface.
Preserves authentic WESAD target semantics: Binary Stress Detection (0 = Non-Stress, 1 = Stress).
"""

import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, List

# Ensure project root is in sys.path
_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import config

class PhysiologicalStressModel:
    """
    Standardized inference wrapper for the WESAD Physiological Model.
    Predicts physiological stress state probability P(Stress=1 | biometrics).
    Does NOT rename physiological stress to threat detection.
    """
    def __init__(self, model_path: Optional[Path] = None):
        self.model_path = model_path or (config.MODELS_DIR / "physiological" / "physiological_model.pkl")
        self.model_name = "Unknown"
        self.model = None
        self.scaler = None
        self.feature_names: List[str] = []
        self.feature_medians: Dict[str, float] = {}
        self.metrics: Dict[str, Any] = {}
        self.is_trained_ml: bool = False
        self.target_semantics = "WESAD Clinical Stress Detection (0=Non-Stress [Baseline/Amusement/Meditation], 1=Stress)"
        self._load_artifact()

    def _load_artifact(self) -> None:
        if not self.model_path.exists():
            print(f"[PhysiologicalStressModel] Warning: Artifact not found at {self.model_path}", flush=True)
            self.is_trained_ml = False
            return

        with open(self.model_path, "rb") as f:
            artifact = pickle.load(f)

        self.model_name = artifact.get("model_name", "RandomForest")
        self.model = artifact.get("model")
        self.scaler = artifact.get("scaler")
        self.feature_names = artifact.get("feature_names", [])
        self.metrics = artifact.get("metrics", {})
        self.is_trained_ml = (self.model is not None)

        # Precompute feature medians from physiological_features.csv if available for imputation
        features_csv = config.DATA_DIR / "physiological" / "physiological_features.csv"
        if features_csv.exists():
            try:
                df = pd.read_csv(features_csv)
                common_cols = [c for c in self.feature_names if c in df.columns]
                self.feature_medians = df[common_cols].median().to_dict()
            except Exception:
                self.feature_medians = {f: 0.0 for f in self.feature_names}
        else:
            self.feature_medians = {f: 0.0 for f in self.feature_names}

    def predict_proba(self, features: Dict[str, float]) -> float:
        """
        Predicts stress probability from biometric feature dictionary.
        Returns float in [0.0, 1.0].
        """
        if not self.is_trained_ml or self.model is None:
            # Fallback heuristic baseline if artifact is missing
            return 0.20

        # Build feature vector matching exact training column ordering
        row = {}
        for fname in self.feature_names:
            val = features.get(fname)
            if val is None or np.isnan(val) or np.isinf(val):
                val = self.feature_medians.get(fname, 0.0)
            row[fname] = float(val)

        X_df = pd.DataFrame([row], columns=self.feature_names)

        if self.scaler is not None:
            X_vec = self.scaler.transform(X_df)
            X_df = pd.DataFrame(X_vec, columns=self.feature_names)

        prob = float(self.model.predict_proba(X_df)[0, 1])
        return min(1.0, max(0.0, prob))

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "branch": "Physiological",
            "model_name": self.model_name,
            "is_trained_ml": self.is_trained_ml,
            "target_semantics": self.target_semantics,
            "num_features": len(self.feature_names),
            "feature_names": self.feature_names,
            "metrics": self.metrics
        }

