import os
import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any

from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
)

# Ensure project root is in sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
import config

def train_and_evaluate_physiological_models(
    features_csv: Path = config.DATA_DIR / "physiological" / "physiological_features.csv",
    output_model_path: Path = config.MODELS_DIR / "physiological" / "physiological_model.pkl"
) -> Dict[str, Any]:
    """
    Loads extracted physiological features, performs GroupKFold cross-validation
    across subject IDs, evaluates multiple algorithms, and saves the best model.
    """
    if not features_csv.exists():
        raise FileNotFoundError(f"Feature dataset not found at {features_csv}. Please run feature extraction first.")
        
    print(f"Loading physiological dataset from: {features_csv}...", flush=True)
    df = pd.read_csv(features_csv)
    print(f"Dataset loaded. Shape: {df.shape}", flush=True)
    
    # Metadata columns vs feature columns
    meta_cols = ["subject_id", "start_sec", "label"]
    feature_cols = [c for c in df.columns if c not in meta_cols]
    
    X = df[feature_cols].copy()
    y = df["label"].values
    groups = df["subject_id"].values
    
    # Handle any potential Inf/NaN values
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())
    
    print(f"Number of feature dimensions: {len(feature_cols)}", flush=True)
    print(f"Unique Subjects (Groups): {len(np.unique(groups))} -> {np.unique(groups).tolist()}", flush=True)
    print(f"Target Distribution: Class 0 (Non-Stress) = {np.sum(y==0)}, Class 1 (Stress) = {np.sum(y==1)}", flush=True)
    
    # Define candidate models
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42, n_jobs=-1),
        "XGBoost": XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, eval_metric="logloss", random_state=42, n_jobs=-1)
    }
    
    # GroupKFold by Subject ID
    gkf = GroupKFold(n_splits=min(5, len(np.unique(groups))))
    
    results = {}
    oof_predictions = {}
    oof_probabilities = {}
    
    for name, model in models.items():
        print(f"\n--- Cross-Validating Model: {name} ---", flush=True)
        oof_pred = np.zeros(len(y))
        oof_prob = np.zeros(len(y))
        
        for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
            X_train, y_train = X.iloc[train_idx], y[train_idx]
            X_val, y_val = X.iloc[val_idx], y[val_idx]
            
            # Scaler fit only on training split to prevent leakage
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            
            if name == "LogisticRegression":
                model.fit(X_train_scaled, y_train)
                pred = model.predict(X_val_scaled)
                prob = model.predict_proba(X_val_scaled)[:, 1]
            else:
                model.fit(X_train, y_train)
                pred = model.predict(X_val)
                prob = model.predict_proba(X_val)[:, 1]
                
            oof_pred[val_idx] = pred
            oof_prob[val_idx] = prob
            
        acc = accuracy_score(y, oof_pred)
        prec = precision_score(y, oof_pred, zero_division=0)
        rec = recall_score(y, oof_pred, zero_division=0)
        f1 = f1_score(y, oof_pred, zero_division=0)
        auc = roc_auc_score(y, oof_prob)
        cm = confusion_matrix(y, oof_pred).tolist()
        
        results[name] = {
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1_score": float(f1),
            "roc_auc": float(auc),
            "confusion_matrix": cm
        }
        
        oof_predictions[name] = oof_pred
        oof_probabilities[name] = oof_prob
        
        print(f"[{name}] Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f} | ROC-AUC: {auc:.4f}", flush=True)

    # Select best model based on ROC-AUC & F1-Score
    best_model_name = max(results.keys(), key=lambda k: results[k]["roc_auc"])
    print(f"\n==========================================")
    print(f"Best Model Selected: {best_model_name} (ROC-AUC: {results[best_model_name]['roc_auc']:.4f})")
    print(f"==========================================")
    
    # Train final best model on all data
    final_scaler = StandardScaler()
    X_scaled_full = final_scaler.fit_transform(X)
    
    best_model_instance = models[best_model_name]
    if best_model_name == "LogisticRegression":
        best_model_instance.fit(X_scaled_full, y)
    else:
        best_model_instance.fit(X, y)
        
    # Save model artifact
    output_model_path.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model_name": best_model_name,
        "model": best_model_instance,
        "scaler": final_scaler if best_model_name == "LogisticRegression" else None,
        "feature_names": feature_cols,
        "metrics": results,
        "oof_probabilities": oof_probabilities[best_model_name],
        "df_metadata": df[["subject_id", "start_sec", "label"]].to_dict(orient="records")
    }
    
    with open(output_model_path, "wb") as f:
        pickle.dump(artifact, f)
        
    print(f"Saved physiological model artifact to: {output_model_path}", flush=True)
    return artifact

if __name__ == "__main__":
    train_and_evaluate_physiological_models()
