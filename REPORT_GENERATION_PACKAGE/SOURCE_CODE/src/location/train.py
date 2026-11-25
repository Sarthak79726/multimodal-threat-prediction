import os
import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

# Ensure project root is in sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
import config
from src.location.generate_data import generate_location_dataset

def train_and_evaluate_location_model(
    data_csv: Path = config.DATA_DIR / "location" / "location_data.csv",
    output_model_path: Path = config.MODELS_DIR / "location" / "location_model.pkl"
) -> Dict[str, Any]:
    """
    Trains and evaluates Model 2 (Location Risk Model).
    """
    if not data_csv.exists():
        print(f"Dataset not found at {data_csv}. Generating synthetic location data...", flush=True)
        generate_location_dataset(output_csv=data_csv)
        
    df = pd.read_csv(data_csv)
    
    X = df.drop(columns=["location_risk_label"])
    y = df["location_risk_label"].values
    
    num_cols = ["latitude", "longitude", "distance_to_security_m", "crime_rate_index", "lighting_level", "isolation_index", "time_of_day_hour"]
    cat_cols = ["location_type"]
    
    # Preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), cat_cols)
        ]
    )
    
    X_processed = preprocessor.fit_transform(X)
    
    # Feature names
    cat_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(cat_cols).tolist()
    feature_names = num_cols + cat_feature_names
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    models = {
        "RandomForest": RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
    }
    if HAS_XGBOOST:
        models["XGBoost"] = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, eval_metric="logloss", random_state=42)
    else:
        models["GradientBoosting"] = GradientBoostingClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
    
    results = {}
    best_model_name = None
    best_auc = 0.0
    best_oof_probs = None
    
    for name, model in models.items():
        oof_preds = np.zeros(len(y))
        oof_probs = np.zeros(len(y))
        
        for train_idx, val_idx in skf.split(X_processed, y):
            X_tr, y_tr = X_processed[train_idx], y[train_idx]
            X_va, y_va = X_processed[val_idx], y[val_idx]
            
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
        
        print(f"Location Model [{name}] Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f} | ROC-AUC: {auc:.4f}", flush=True)
        
        if auc > best_auc:
            best_auc = auc
            best_model_name = name
            best_oof_probs = oof_probs
            
    # Retrain best model on all data
    final_model = models[best_model_name]
    final_model.fit(X_processed, y)
    
    output_model_path.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model_name": best_model_name,
        "model": final_model,
        "preprocessor": preprocessor,
        "feature_names": feature_names,
        "metrics": results,
        "oof_probabilities": best_oof_probs
    }
    
    with open(output_model_path, "wb") as f:
        pickle.dump(artifact, f)
        
    print(f"Saved location model artifact ({best_model_name}) to: {output_model_path}", flush=True)
    return artifact

if __name__ == "__main__":
    train_and_evaluate_location_model()
