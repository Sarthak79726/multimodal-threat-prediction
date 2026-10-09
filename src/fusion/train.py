import os
import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any

from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

# Ensure project root is in sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
import config
from src.fusion.fusion_dataset import generate_fusion_dataset

def train_and_evaluate_fusion_model(
    data_csv: Path = config.DATA_DIR / "fusion" / "fusion_data.csv",
    output_model_path: Path = config.MODELS_DIR / "fusion" / "fusion_model.pkl"
) -> Dict[str, Any]:
    """
    Trains and evaluates Meta-Learner Fusion Model combining 3 input risk scores:
    [physiological_stress_score, location_risk_score, environment_risk_score]
    """
    # Always regenerate fusion dataset to match 3-branch architecture
    generate_fusion_dataset(output_csv=data_csv)
        
    df = pd.read_csv(data_csv)
    
    feature_cols = ["physiological_stress_score", "location_risk_score", "environment_risk_score"]
    X = df[feature_cols].values
    y = df["threat_label"].values
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    models = {
        "LogisticRegression_Stacker": LogisticRegression(C=1.0, random_state=42),
        "RandomForest_Stacker": RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    }
    if HAS_XGBOOST:
        models["XGBoost_Stacker"] = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, eval_metric="logloss", random_state=42)
    else:
        models["GradientBoosting_Stacker"] = GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42)
    
    results = {}
    best_model_name = None
    best_auc = 0.0
    
    for name, model in models.items():
        oof_preds = np.zeros(len(y))
        oof_probs = np.zeros(len(y))
        
        for train_idx, val_idx in skf.split(X, y):
            X_tr, y_tr = X[train_idx], y[train_idx]
            X_va, y_va = X[val_idx], y[val_idx]
            
            model.fit(X_tr, y_tr)
            preds = model.predict(X_va)
            probs = model.predict_proba(X_va)[:, 1]
            
            oof_preds[val_idx] = preds
            oof_probs[val_idx] = probs
            
        acc = accuracy_score(y, oof_preds)
        prec = precision_score(y, oof_preds)
        rec = recall_score(y, oof_preds)
        f1 = f1_score(y, oof_preds)
        auc = roc_auc_score(y, oof_probs)
        
        results[name] = {
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1_score": float(f1),
            "roc_auc": float(auc),
            "confusion_matrix": confusion_matrix(y, oof_preds).tolist()
        }
        
        print(f"Fusion Model [{name}] Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f} | ROC-AUC: {auc:.4f}", flush=True)
        
        if auc > best_auc:
            best_auc = auc
            best_model_name = name
            
    final_model = models[best_model_name]
    final_model.fit(X, y)
    
    # Feature importances / weights
    if hasattr(final_model, "coef_"):
        weights = dict(zip(feature_cols, final_model.coef_[0].tolist()))
    elif hasattr(final_model, "feature_importances_"):
        weights = dict(zip(feature_cols, final_model.feature_importances_.tolist()))
    else:
        weights = {c: 0.333 for c in feature_cols}
        
    output_model_path.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model_name": best_model_name,
        "model": final_model,
        "feature_names": feature_cols,
        "weights": weights,
        "metrics": results
    }
    
    with open(output_model_path, "wb") as f:
        pickle.dump(artifact, f)
        
    print(f"Saved Fusion Model artifact ({best_model_name}) to: {output_model_path}", flush=True)
    return artifact

if __name__ == "__main__":
    train_and_evaluate_fusion_model()
