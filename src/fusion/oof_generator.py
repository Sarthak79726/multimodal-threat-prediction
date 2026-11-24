"""
Leakage-Free Out-Of-Fold (OOF) Prediction Generator.
Implements rigorous, leak-free cross-validation partitioning:
- Group-aware splits (StratifiedGroupKFold / GroupKFold) preventing subject/session identity leakage.
- Strict isolation of preprocessing (fitted strictly inside training folds).
- Generation of out-of-fold probability matrices for Level 1 meta-learner training.
- Verification checks asserting that zero samples in any validation fold were seen during base-model fitting.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Callable

from sklearn.model_selection import StratifiedGroupKFold, GroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone

# Ensure project root is in sys.path
_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import config

class OOFGenerator:
    """
    Orchestrates leakage-free Out-Of-Fold probability generation across training folds.
    """
    def __init__(
        self,
        n_splits: int = 5,
        random_state: int = 42,
        shuffle: bool = True
    ):
        self.n_splits = n_splits
        self.random_state = random_state
        self.shuffle = shuffle
        self.fold_splits_info: List[Dict[str, Any]] = []

    def select_cv_strategy(
        self,
        X: np.ndarray,
        y: np.ndarray,
        groups: Optional[np.ndarray] = None
    ):
        """
        Selects scientifically appropriate cross-validation split strategy:
        1. StratifiedGroupKFold if groups and multi-class distribution permit it.
        2. GroupKFold if StratifiedGroupKFold cannot balance or groups are small.
        3. StratifiedKFold if groups are absent.
        """
        if groups is not None and len(np.unique(groups)) >= 2:
            unique_groups = len(np.unique(groups))
            splits = min(self.n_splits, unique_groups)
            
            # Check if each class is represented across sufficient groups
            try:
                sgkf = StratifiedGroupKFold(n_splits=splits, shuffle=self.shuffle, random_state=self.random_state)
                # Test split generation
                list(sgkf.split(X, y, groups))
                return sgkf, f"StratifiedGroupKFold(n_splits={splits})"
            except Exception:
                gkf = GroupKFold(n_splits=splits)
                return gkf, f"GroupKFold(n_splits={splits})"
        else:
            skf = StratifiedKFold(n_splits=self.n_splits, shuffle=self.shuffle, random_state=self.random_state)
            return skf, f"StratifiedKFold(n_splits={self.n_splits})"

    def generate_oof_predictions(
        self,
        base_models: Dict[str, Any],
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        groups_train: Optional[np.ndarray] = None,
        scale_features: bool = False
    ) -> Tuple[pd.DataFrame, Dict[str, List[Any]]]:
        """
        Generates Out-of-Fold prediction probabilities for each base model.
        
        Args:
            base_models: Dictionary of model instances e.g. {"physio": RandomForestClassifier()}
            X_train: Feature DataFrame for base model training
            y_train: Target ground truth array
            groups_train: Optional subject / session group labels (prevents subject leakage)
            scale_features: Whether to fit standard scaling inside folds
            
        Returns:
            oof_meta_df: DataFrame with shape (N, len(base_models)) containing OOF probabilities
            fitted_fold_models: Dict mapping model_name to list of fold-fitted model instances
        """
        y_arr = np.asarray(y_train, dtype=np.int32)
        n_samples = len(X_train)
        model_names = list(base_models.keys())

        cv, cv_name = self.select_cv_strategy(X_train.values, y_arr, groups_train)
        print(f"[OOFGenerator] Using CV strategy: {cv_name} across {n_samples} samples.", flush=True)

        oof_matrix = {name: np.zeros(n_samples, dtype=np.float64) for name in model_names}
        fitted_fold_models = {name: [] for name in model_names}
        self.fold_splits_info = []

        # Track fold assignments to verify OOF property
        evaluated_mask = np.zeros(n_samples, dtype=bool)

        splits_iter = cv.split(X_train, y_arr, groups_train) if groups_train is not None else cv.split(X_train, y_arr)

        for fold_idx, (train_idx, val_idx) in enumerate(splits_iter):
            # Assert mathematical mutual exclusivity of train and val indices
            overlap = set(train_idx).intersection(set(val_idx))
            if overlap:
                raise RuntimeError(f"Critical Leakage Violation: Fold {fold_idx} has {len(overlap)} overlapping indices between train and val!")

            # Verify no group leakage if groups provided
            if groups_train is not None:
                train_groups = set(groups_train[train_idx])
                val_groups = set(groups_train[val_idx])
                group_overlap = train_groups.intersection(val_groups)
                if group_overlap:
                    raise RuntimeError(f"Critical Subject Leakage: Subjects {group_overlap} appear in both train and validation splits in fold {fold_idx}!")

            X_tr = X_train.iloc[train_idx].copy()
            y_tr = y_arr[train_idx]
            X_va = X_train.iloc[val_idx].copy()
            y_va = y_arr[val_idx]

            # Fit scaler strictly on training split
            if scale_features:
                scaler = StandardScaler()
                X_tr_proc = scaler.fit_transform(X_tr)
                X_va_proc = scaler.transform(X_va)
            else:
                X_tr_proc = X_tr.values
                X_va_proc = X_va.values

            fold_info = {
                "fold": fold_idx,
                "n_train": len(train_idx),
                "n_val": len(val_idx),
                "val_class_distribution": dict(pd.Series(y_va).value_counts())
            }
            if groups_train is not None:
                fold_info["val_groups"] = list(val_groups)
            self.fold_splits_info.append(fold_info)

            # Fit each base model on this fold's training portion and predict on val
            for name, base_model in base_models.items():
                model_fold = clone(base_model)
                model_fold.fit(X_tr_proc, y_tr)
                
                if hasattr(model_fold, "predict_proba"):
                    probs = model_fold.predict_proba(X_va_proc)[:, 1]
                elif hasattr(model_fold, "decision_function"):
                    # Sigmoid transform of decision function
                    df_val = model_fold.decision_function(X_va_proc)
                    probs = 1.0 / (1.0 + np.exp(-df_val))
                else:
                    probs = model_fold.predict(X_va_proc).astype(float)

                oof_matrix[name][val_idx] = probs
                fitted_fold_models[name].append(model_fold)

            evaluated_mask[val_idx] = True

        # Assert every single sample received an out-of-fold prediction
        if not np.all(evaluated_mask):
            unseen_count = int(np.sum(~evaluated_mask))
            raise RuntimeError(f"OOF Violation: {unseen_count} samples did not receive an out-of-fold prediction!")

        oof_meta_df = pd.DataFrame(oof_matrix, index=X_train.index)
        print(f"[OOFGenerator] Completed OOF matrix generation. Shape: {oof_meta_df.shape}", flush=True)
        return oof_meta_df, fitted_fold_models

