"""
Comprehensive Training and Evaluation Pipeline for Environmental Models.
Trains two separate models:
1. Model A: Crowd Density Prediction (Target: simulated_crowd_density_category -> Low, Medium, High)
2. Model B: Lighting Condition Prediction (Target: simulated_lighting_category -> Dark, Low, Good)
   (Plus optional regression for simulated_lighting_lux_estimate)

Strictly uses chronological Train / Validation / Test splits:
- Train: Days 1-120 (48,960 samples)
- Validation: Days 121-150 (12,240 samples)
- Test: Days 151-180 (12,240 samples)

Preprocessors are fitted strictly on Train split only.
"""

import sys
import json
import pickle
import datetime
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Tuple

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, mean_squared_error, mean_absolute_error, r2_score
)

_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import config

MODELS_DIR = config.MODELS_DIR / "environment"
REPORTS_DIR = Path(_project_root) / "reports" / "environmental_models"
DATA_PATH = config.DATA_DIR / "environment" / "BIT_Mesra_Synthetic_Environmental_Dataset.csv"

FEATURE_COLS = ["location", "hour_of_day", "day_of_week", "weekend_indicator"]
CAT_COLS = ["location", "day_of_week"]
NUM_COLS = ["hour_of_day", "weekend_indicator"]

def df_to_markdown(df: pd.DataFrame) -> str:
    """Formats a DataFrame as a Markdown table without requiring tabulate."""
    headers = [str(col) for col in df.columns]
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in df.columns) + " |")
    return "\n".join(lines)

def build_preprocessor() -> ColumnTransformer:
    """Creates standardized preprocessor for environmental features."""
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_COLS),
            ("num", StandardScaler(), NUM_COLS)
        ]
    )

def train_and_evaluate_environmental_models() -> Dict[str, Any]:
    """
    Executes full training, validation, selection, and testing pipeline
    for both Crowd Density and Lighting Condition models.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("ENVIRONMENTAL AI MODEL TRAINING PIPELINE")
    print(f"Dataset: {DATA_PATH}")
    print("=" * 80)

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Synthetic dataset not found at: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    print(f"Loaded dataset with shape {df.shape}")

    # Chronological Split
    df_train = df[df["split"] == "Train"].copy()
    df_val = df[df["split"] == "Validation"].copy()
    df_test = df[df["split"] == "Test"].copy()

    print(f"Split sizes - Train: {len(df_train)}, Validation: {len(df_val)}, Test: {len(df_test)}")

    # 1. Feature Preprocessing (fit strictly on Train)
    preprocessor = build_preprocessor()
    X_train_raw = df_train[FEATURE_COLS]
    X_val_raw = df_val[FEATURE_COLS]
    X_test_raw = df_test[FEATURE_COLS]

    X_train = preprocessor.fit_transform(X_train_raw)
    X_val = preprocessor.transform(X_val_raw)
    X_test = preprocessor.transform(X_test_raw)

    cat_feature_names = preprocessor.named_transformers_["cat"].get_feature_names_out(CAT_COLS).tolist()
    feature_names = cat_feature_names + NUM_COLS
    print(f"Engineered {len(feature_names)} features after OneHotEncoding.")

    # =========================================================================
    # MODEL A: CROWD DENSITY PREDICTION
    # Classes: Low, Medium, High
    # =========================================================================
    print("\n" + "=" * 80)
    print("TRAINING MODEL A: CROWD DENSITY PREDICTION")
    print("Target: simulated_crowd_density_category ('Low', 'Medium', 'High')")
    print("=" * 80)

    target_col_crowd = "simulated_crowd_density_category"
    crowd_classes = ["Low", "Medium", "High"]
    class_to_idx_crowd = {c: i for i, c in enumerate(crowd_classes)}
    idx_to_class_crowd = {i: c for i, c in enumerate(crowd_classes)}

    y_train_crowd = np.array([class_to_idx_crowd[val] for val in df_train[target_col_crowd]])
    y_val_crowd = np.array([class_to_idx_crowd[val] for val in df_val[target_col_crowd]])
    y_test_crowd = np.array([class_to_idx_crowd[val] for val in df_test[target_col_crowd]])

    crowd_candidate_models = {
        "Dummy Classifier": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1),
        "Gradient Boosting (XGBoost)": XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, eval_metric="mlogloss")
    }

    crowd_results = []
    best_crowd_model_name = None
    best_crowd_val_f1 = -1.0
    fitted_crowd_models = {}

    for name, model in crowd_candidate_models.items():
        print(f"--> Training Model A Candidate: {name}...")
        model.fit(X_train, y_train_crowd)
        fitted_crowd_models[name] = model

        # Val evaluation
        val_preds = model.predict(X_val)
        val_acc = accuracy_score(y_val_crowd, val_preds)
        val_prec = precision_score(y_val_crowd, val_preds, average="macro", zero_division=0)
        val_rec = recall_score(y_val_crowd, val_preds, average="macro", zero_division=0)
        val_f1 = f1_score(y_val_crowd, val_preds, average="macro", zero_division=0)

        # Test evaluation
        test_preds = model.predict(X_test)
        test_acc = accuracy_score(y_test_crowd, test_preds)
        test_prec = precision_score(y_test_crowd, test_preds, average="macro", zero_division=0)
        test_rec = recall_score(y_test_crowd, test_preds, average="macro", zero_division=0)
        test_f1 = f1_score(y_test_crowd, test_preds, average="macro", zero_division=0)

        crowd_results.append({
            "Model": name,
            "Val_Accuracy": round(val_acc, 4),
            "Val_Macro_Precision": round(val_prec, 4),
            "Val_Macro_Recall": round(val_rec, 4),
            "Val_Macro_F1": round(val_f1, 4),
            "Test_Accuracy": round(test_acc, 4),
            "Test_Macro_Precision": round(test_prec, 4),
            "Test_Macro_Recall": round(test_rec, 4),
            "Test_Macro_F1": round(test_f1, 4)
        })

        print(f"    Val F1: {val_f1:.4f} | Test F1: {test_f1:.4f} | Test Acc: {test_acc:.4f}")

        if val_f1 > best_crowd_val_f1:
            best_crowd_val_f1 = val_f1
            best_crowd_model_name = name

    print(f"\n[Model Selection] Best Crowd Model: {best_crowd_model_name} (Val Macro F1: {best_crowd_val_f1:.4f})")
    best_crowd_model = fitted_crowd_models[best_crowd_model_name]
    best_crowd_test_preds = best_crowd_model.predict(X_test)
    crowd_cm = confusion_matrix(y_test_crowd, best_crowd_test_preds)
    crowd_report = classification_report(y_test_crowd, best_crowd_test_preds, target_names=crowd_classes, output_dict=True)

    # =========================================================================
    # MODEL B: LIGHTING CONDITION PREDICTION
    # Classes: Dark, Low, Good
    # =========================================================================
    print("\n" + "=" * 80)
    print("TRAINING MODEL B: LIGHTING CONDITION PREDICTION")
    print("Target: simulated_lighting_category ('Dark', 'Low', 'Good')")
    print("=" * 80)

    target_col_light = "simulated_lighting_category"
    light_classes = ["Dark", "Low", "Good"]
    class_to_idx_light = {c: i for i, c in enumerate(light_classes)}
    idx_to_class_light = {i: c for i, c in enumerate(light_classes)}

    y_train_light = np.array([class_to_idx_light[val] for val in df_train[target_col_light]])
    y_val_light = np.array([class_to_idx_light[val] for val in df_val[target_col_light]])
    y_test_light = np.array([class_to_idx_light[val] for val in df_test[target_col_light]])

    light_candidate_models = {
        "Dummy Classifier": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1),
        "Gradient Boosting (XGBoost)": XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, eval_metric="mlogloss")
    }

    light_results = []
    best_light_model_name = None
    best_light_val_f1 = -1.0
    fitted_light_models = {}

    for name, model in light_candidate_models.items():
        print(f"--> Training Model B Candidate: {name}...")
        model.fit(X_train, y_train_light)
        fitted_light_models[name] = model

        # Val evaluation
        val_preds = model.predict(X_val)
        val_acc = accuracy_score(y_val_light, val_preds)
        val_prec = precision_score(y_val_light, val_preds, average="macro", zero_division=0)
        val_rec = recall_score(y_val_light, val_preds, average="macro", zero_division=0)
        val_f1 = f1_score(y_val_light, val_preds, average="macro", zero_division=0)

        # Test evaluation
        test_preds = model.predict(X_test)
        test_acc = accuracy_score(y_test_light, test_preds)
        test_prec = precision_score(y_test_light, test_preds, average="macro", zero_division=0)
        test_rec = recall_score(y_test_light, test_preds, average="macro", zero_division=0)
        test_f1 = f1_score(y_test_light, test_preds, average="macro", zero_division=0)

        light_results.append({
            "Model": name,
            "Val_Accuracy": round(val_acc, 4),
            "Val_Macro_Precision": round(val_prec, 4),
            "Val_Macro_Recall": round(val_rec, 4),
            "Val_Macro_F1": round(val_f1, 4),
            "Test_Accuracy": round(test_acc, 4),
            "Test_Macro_Precision": round(test_prec, 4),
            "Test_Macro_Recall": round(test_rec, 4),
            "Test_Macro_F1": round(test_f1, 4)
        })

        print(f"    Val F1: {val_f1:.4f} | Test F1: {test_f1:.4f} | Test Acc: {test_acc:.4f}")

        if val_f1 > best_light_val_f1:
            best_light_val_f1 = val_f1
            best_light_model_name = name

    print(f"\n[Model Selection] Best Lighting Model: {best_light_model_name} (Val Macro F1: {best_light_val_f1:.4f})")
    best_light_model = fitted_light_models[best_light_model_name]
    best_light_test_preds = best_light_model.predict(X_test)
    light_cm = confusion_matrix(y_test_light, best_light_test_preds)
    light_report = classification_report(y_test_light, best_light_test_preds, target_names=light_classes, output_dict=True)

    # =========================================================================
    # OPTIONAL EXPERIMENT: REGRESSION FOR LUX ESTIMATION
    # =========================================================================
    print("\n" + "=" * 80)
    print("EXPERIMENTAL REGRESSION: LIGHTING LUX ESTIMATION")
    print("Target: simulated_lighting_lux_estimate (Continuous)")
    print("=" * 80)

    y_train_lux = df_train["simulated_lighting_lux_estimate"].values
    y_val_lux = df_val["simulated_lighting_lux_estimate"].values
    y_test_lux = df_test["simulated_lighting_lux_estimate"].values

    lux_regressor = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    lux_regressor.fit(X_train, y_train_lux)
    val_lux_preds = lux_regressor.predict(X_val)
    test_lux_preds = lux_regressor.predict(X_test)

    lux_metrics = {
        "Val_RMSE": round(float(np.sqrt(mean_squared_error(y_val_lux, val_lux_preds))), 2),
        "Val_MAE": round(float(mean_absolute_error(y_val_lux, val_lux_preds)), 2),
        "Val_R2": round(float(r2_score(y_val_lux, val_lux_preds)), 4),
        "Test_RMSE": round(float(np.sqrt(mean_squared_error(y_test_lux, test_lux_preds))), 2),
        "Test_MAE": round(float(mean_absolute_error(y_test_lux, test_lux_preds)), 2),
        "Test_R2": round(float(r2_score(y_test_lux, test_lux_preds)), 4)
    }
    print(f"Lux Regressor Test RMSE: {lux_metrics['Test_RMSE']} | MAE: {lux_metrics['Test_MAE']} | R2: {lux_metrics['Test_R2']}")

    # =========================================================================
    # SAVE SERIALIZED MODEL ARTIFACTS WITH METADATA
    # =========================================================================
    train_timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    crowd_artifact = {
        "model_name": best_crowd_model_name,
        "estimator": best_crowd_model,
        "preprocessor": preprocessor,
        "classes": crowd_classes,
        "class_to_idx": class_to_idx_crowd,
        "idx_to_class": idx_to_class_crowd,
        "feature_names": feature_names,
        "input_features": FEATURE_COLS,
        "metadata": {
            "dataset_name": "BIT_Mesra_Synthetic_Environmental_Dataset",
            "dataset_version": "1.0_synthetic_180days",
            "synthetic_provenance": "SYNTHETIC_SIMULATION_RULE_GENERATED",
            "data_disclaimer": "SYNTHETIC-DATA TRAINED MODEL — EXPERIMENTAL PREDICTION — NOT LIVE SENSOR MEASUREMENT",
            "training_date": train_timestamp,
            "training_samples": len(df_train),
            "validation_samples": len(df_val),
            "test_samples": len(df_test),
            "target": target_col_crowd,
            "validation_macro_f1": best_crowd_val_f1,
            "test_macro_f1": crowd_results[[r["Model"] for r in crowd_results].index(best_crowd_model_name)]["Test_Macro_F1"],
            "test_accuracy": crowd_results[[r["Model"] for r in crowd_results].index(best_crowd_model_name)]["Test_Accuracy"]
        }
    }

    crowd_model_path = MODELS_DIR / "crowd_model.pkl"
    with open(crowd_model_path, "wb") as f:
        pickle.dump(crowd_artifact, f)
    print(f"\n[Artifact Saved] Crowd Model -> {crowd_model_path}")

    lighting_artifact = {
        "model_name": best_light_model_name,
        "estimator": best_light_model,
        "preprocessor": preprocessor,
        "classes": light_classes,
        "class_to_idx": class_to_idx_light,
        "idx_to_class": idx_to_class_light,
        "feature_names": feature_names,
        "input_features": FEATURE_COLS,
        "lux_regressor": lux_regressor,
        "metadata": {
            "dataset_name": "BIT_Mesra_Synthetic_Environmental_Dataset",
            "dataset_version": "1.0_synthetic_180days",
            "synthetic_provenance": "SYNTHETIC_SIMULATION_RULE_GENERATED",
            "data_disclaimer": "SYNTHETIC-DATA TRAINED MODEL — EXPERIMENTAL PREDICTION — NOT LIVE SENSOR MEASUREMENT",
            "training_date": train_timestamp,
            "training_samples": len(df_train),
            "validation_samples": len(df_val),
            "test_samples": len(df_test),
            "target": target_col_light,
            "validation_macro_f1": best_light_val_f1,
            "test_macro_f1": light_results[[r["Model"] for r in light_results].index(best_light_model_name)]["Test_Macro_F1"],
            "test_accuracy": light_results[[r["Model"] for r in light_results].index(best_light_model_name)]["Test_Accuracy"],
            "lux_metrics": lux_metrics
        }
    }

    lighting_model_path = MODELS_DIR / "lighting_model.pkl"
    with open(lighting_model_path, "wb") as f:
        pickle.dump(lighting_artifact, f)
    print(f"[Artifact Saved] Lighting Model -> {lighting_model_path}")

    # =========================================================================
    # GENERATE BENCHMARK COMPARISON REPORTS
    # =========================================================================
    df_crowd_res = pd.DataFrame(crowd_results)
    df_light_res = pd.DataFrame(light_results)

    df_crowd_res.to_csv(REPORTS_DIR / "crowd_model_comparison.csv", index=False)
    df_light_res.to_csv(REPORTS_DIR / "lighting_model_comparison.csv", index=False)

    # Save Markdown Evaluation Report
    report_md_path = REPORTS_DIR / "environmental_model_evaluation.md"
    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write("# Environmental Context AI Models Evaluation Report\n\n")
        f.write(f"**Generated:** {train_timestamp}  \n")
        f.write("**Status:** EXPERIMENTAL PREDICTION MODELS (TRAINED ON SYNTHETIC DATA)  \n")
        f.write("**Warning:** SYNTHETIC-DATA TRAINED MODEL — NOT LIVE SENSOR MEASUREMENTS  \n\n")

        f.write("## 1. Dataset & Split Specification\n\n")
        f.write("- **Dataset:** `BIT_Mesra_Synthetic_Environmental_Dataset.xlsx` / `.csv`\n")
        f.write(f"- **Total Samples:** {len(df):,} records\n")
        f.write(f"- **Train Set (Days 1–120):** {len(df_train):,} records (Chronological split)\n")
        f.write(f"- **Validation Set (Days 121–150):** {len(df_val):,} records\n")
        f.write(f"- **Test Set (Days 151–180):** {len(df_test):,} records\n")
        f.write(f"- **Locations:** 17 BIT Mesra campus landmarks\n")
        f.write(f"- **Features:** `location`, `hour_of_day`, `day_of_week`, `weekend_indicator`\n\n")

        f.write("## 2. Model A: Crowd Density Prediction Comparison\n\n")
        f.write(df_to_markdown(df_crowd_res) + "\n\n")
        f.write(f"**Selected Model:** `{best_crowd_model_name}`  \n")
        f.write(f"**Test Confusion Matrix (Rows=True, Cols=Pred: Low, Medium, High):**\n\n")
        f.write("```\n" + str(crowd_cm) + "\n```\n\n")
        f.write("### Per-Class Test Performance (Model A)\n\n")
        f.write(df_to_markdown(pd.DataFrame(crowd_report).transpose().reset_index()) + "\n\n")

        f.write("## 3. Model B: Lighting Condition Prediction Comparison\n\n")
        f.write(df_to_markdown(df_light_res) + "\n\n")
        f.write(f"**Selected Model:** `{best_light_model_name}`  \n")
        f.write(f"**Test Confusion Matrix (Rows=True, Cols=Pred: Dark, Low, Good):**\n\n")
        f.write("```\n" + str(light_cm) + "\n```\n\n")
        f.write("### Per-Class Test Performance (Model B)\n\n")
        f.write(df_to_markdown(pd.DataFrame(light_report).transpose().reset_index()) + "\n\n")

        f.write("## 4. Experimental Continuous Lux Estimation (Regression)\n\n")
        f.write(f"- **Model:** Random Forest Regressor (100 trees, max_depth=10)\n")
        f.write(f"- **Test RMSE:** {lux_metrics['Test_RMSE']} lux\n")
        f.write(f"- **Test MAE:** {lux_metrics['Test_MAE']} lux\n")
        f.write(f"- **Test R² Score:** {lux_metrics['Test_R2']}\n")
        f.write("*(Note: Lux target is synthetically simulated and must not be interpreted as physical photometer measurements)*\n\n")

        f.write("## 5. Methodological Limitations & Transparency\n\n")
        f.write("1. **Direct Generator Rule Encoding:** Because the synthetic dataset was generated using campus baseline schedule rules, high model scores indicate that the algorithms have effectively learned the deterministic rule mappings rather than empirical physical reality.\n")
        f.write("2. **Absence of Real-time Optical/Acoustic Sensors:** The model predicts expected schedule conditions based on campus geography and calendar clock time. It cannot detect dynamic impromptu gatherings, power grid blackouts, or adverse weather conditions.\n")

    print(f"[Report Generated] Evaluation Report -> {report_md_path}")

    return {
        "best_crowd_model": best_crowd_model_name,
        "best_light_model": best_light_model_name,
        "crowd_results": crowd_results,
        "light_results": light_results,
        "lux_metrics": lux_metrics
    }

if __name__ == "__main__":
    train_and_evaluate_environmental_models()
