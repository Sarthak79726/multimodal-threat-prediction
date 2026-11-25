import os
import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

# Ensure project root is in sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
import config
from src.context.generate_data import generate_context_dataset

def train_and_evaluate_context_model(
    data_csv: Path = config.DATA_DIR / "context" / "context_data.csv",
    output_model_path: Path = config.MODELS_DIR / "context" / "context_model.pkl"
) -> Dict[str, Any]:
    """
    Trains and evaluates Model 3 (Context Risk Model).
    """
    if not data_csv.exists():
        print(f"Dataset not found at {data_csv}. Generating synthetic context data...", flush=True)
        generate_context_dataset(output_csv=data_csv)
        
    df = pd.read_csv(data_csv)
    
    X = df.drop(columns=["context_risk_label"])
    y = df["context_risk_label"].values
    feature_names = list(X.columns)
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    models = {
        "RandomForest": RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42),
        "XGBoost": XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, eval_metric="logloss", random_state=42)
    }
    
    results = {}
    best_model_name = None
    best_auc = 0.0
    best_oof_probs = None
    
    for name, model in models.items():
        oof_preds = np.zeros(len(y))
        oof_probs = np.zeros(len(y))
        
        for train_idx, val_idx in skf.split(X_scaled, y):
            X_tr, y_tr = X_scaled[train_idx], y[train_idx]
            X_va, y_va = X_scaled[val_idx], y[val_idx]
            
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
        
        print(f"Context Model [{name}] Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f} | ROC-AUC: {auc:.4f}", flush=True)
        
        if auc > best_auc:
            best_auc = auc
            best_model_name = name
            best_oof_probs = oof_probs
            
    final_model = models[best_model_name]
    final_model.fit(X_scaled, y)
    
    output_model_path.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model_name": best_model_name,
        "model": final_model,
        "scaler": scaler,
        "feature_names": feature_names,
        "metrics": results,
        "oof_probabilities": best_oof_probs
    }
    
    with open(output_model_path, "wb") as f:
        pickle.dump(artifact, f)
        
    print(f"Saved context model artifact ({best_model_name}) to: {output_model_path}", flush=True)
    return artifact

if __name__ == "__main__":
    train_and_evaluate_context_model()
