# 🏛️ Technical Architecture: Multimodal Threat Prediction Engine

**Project:** Multimodal Threat Prediction Engine (B.Tech AIML Project)  
**Deployment Context:** Birla Institute of Technology (BIT) Mesra Campus, Ranchi, India  
**Production Meta-Model Status:** `UNTRAINED — VALID MULTIMODAL THREAT DATASET REQUIRED`  
**Active Production Index:** `Experimental Weighted Context Index` (Heuristic Baseline)

---

## 1. System Overview

The **Multimodal Threat Prediction Engine** is designed to explore the integration of three distinct analytical streams to evaluate personal situational context on the BIT Mesra university campus:

1. **Physiological Analysis:** Wearable sensor biometric signals.
2. **Geospatial Analysis:** Live GPS positioning evaluated against campus geography.
3. **Environmental Analysis:** Contextual situational parameters (time, ambient illumination, crowd density) automatically evaluated using trained Machine Learning models (Model A: XGBoost Crowd Predictor, Model B: Random Forest Lighting Predictor) with automatic Asia/Kolkata system time and GPS coordinate landmark resolution.

These three analytical streams constitute the **Level 0 Base Input Branches**. The architecture incorporates a **Level 1 Machine Learning Stacking Ensemble** designed to fuse the branch outputs into an optimal multimodal probability estimate once a verified multimodal training dataset becomes available.

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                         LIVE CLIENT TELEMETRY / SENSORS                          │
└────────────────────────────────────────┬─────────────────────────────────────────┘
                                         │
         ┌───────────────────────────────┼───────────────────────────────┐
         │                               │                               │
         ▼                               ▼                               ▼
┌─────────────────────────────┐ ┌─────────────────────────────┐ ┌─────────────────────────────┐
│ 1. PHYSIOLOGICAL BRANCH     │ │ 2. SPATIAL BRANCH           │ │ 3. ENVIRONMENTAL BRANCH     │
│ WESAD Wearable Biometrics   │ │ BIT Mesra Campus GPS        │ │ AI-Based Environmental Pred.│
│ • Chest: ECG, EDA, EMG, TEMP│ │ • HTML5 Browser GPS         │ │ • Auto Asia/Kolkata (IST)   │
│ • Wrist: BVP, EDA, TEMP, ACC│ │ • 17 Verified BIT Landmarks │ │ • Auto GPS Geofence Match   │
│ • 76 Extracted Features     │ │ • Security Post Distances   │ │ • 73,440 Synthetic Records  │
│                             │ │                             │ │                             │
│ Model: Random Forest        │ │ Model: Deterministic Haversine│ Model A: XGBoost (Crowd)    │
│ Output: P(Stress|x) ∈ [0,1] │ │ Output: P_spatial ∈ [0,1]   │ │ Model B: Random Forest(Light│
│ STATUS: Trained ML (WESAD)  │ │ STATUS: Deterministic Engine│ │ STATUS: Trained ML (Synth)  │
└──────────────┬──────────────┘ └──────────────┬──────────────┘ └──────────────┬──────────────┘
               │                               │                               │
               └───────────────────────┬───────┴───────────────────────────────┘
                                       │
                                       ▼
                       ┌────────────────────────────────┐
                       │   META-FEATURE VECTOR X_meta   │
                       │   [P_physio, P_spatial, P_env] │
                       └───────────────┬────────────────┘
                                       │
                ┌──────────────────────┴──────────────────────┐
                ▼                                             ▼
┌──────────────────────────────────────┐    ┌──────────────────────────────────────┐
│ PRODUCTION INFERENCE ENGINE          │    │ ISOLATED SYNTHETIC SIMULATION DEMO   │
│ (src/fusion/predict.py)              │    │ (models/fusion/synthetic_demo/)      │
│                                      │    │                                      │
│ • Meta-Learner: UNTRAINED            │    │ • Pipeline Mechanics Demonstration   │
│ • Production Artifacts: REJECTED     │    │ • Stacking Classifier Demo Artifact  │
│   if marked synthetic                │    │ • 4,000 Synthetic Beta Samples      │
│ • Active Calculation:                │    │ • Verification of OOF Cross-Val      │
│   Experimental Weighted Context      │    │ • STRICTLY ISOLATED FROM PRODUCTION  │
│   Index (40% Phys / 25% Loc / 35% Env│    │                                      │
│ • Status: STANDARD CONTEXT /         │    │ Location:                            │
│   ELEVATED CONTEXT RISK              │    │ models/fusion/synthetic_demo/        │
│ • Notice: Heuristic Context Score    │    │ reports/synthetic_simulation/        │
└──────────────────────────────────────┘    └──────────────────────────────────────┘
```

---

## 2. Branch-by-Branch Technical Specifications

### Branch 1: Physiological Stress Detection Model
- **Scientific Goal:** Estimate human sympathetic nervous system arousal and stress state from physiological signals.
- **Dataset:** WESAD (Wearable Stress and Affect Detection), an established multimodal laboratory benchmark collected from 15 subjects under controlled physiological stress induction (Trier Social Stress Test).
- **Sensors:** RespiBAN chest harness (ECG, EDA, EMG, Respiration, Temperature, 3-axis Accelerometer) and Empatica E4 wristband (BVP, EDA, Temperature, 3-axis Accelerometer).
- **Feature Extraction:** 76 statistical, temporal, and frequency-domain features computed over 60-second sliding windows (HRV indices, RMSSD, pNN50, LF/HF ratio, EDA phasic/tonic components, SCR peak count).
- **Implementation:** `src/physiological/model.py`, `PhysiologicalStressModel`.
- **Underlying Estimator:** `RandomForestClassifier` trained with subject-grouped cross-validation.
- **Output Semantics:** Continuous probability $P(\text{Stress} = 1 \mid \mathbf{x}) \in [0.0, 1.0]$.
- **Academic Precision:** This model detects **physiological stress**, not physical danger or threat events. Stress is an associative biomarker that can indicate fear, but also cognitive exertion, physical exercise, or excitement.

### Branch 2: Spatial Risk Engine
- **Scientific Goal:** Assess spatial vulnerability of a client relative to known campus infrastructure at BIT Mesra.
- **Implementation:** `src/location/model.py`, `SpatialRiskEngine`, and `src/location/campus_map.py`, `CampusMapAnalyzer`.
- **Algorithm:** Deterministic geodesic computation using the great-circle Haversine formula relative to 17 verified BIT Mesra campus landmarks.
- **Features Evaluated:**
  - Great-circle distance to nearest campus landmark ($d_{\text{landmark}}$).
  - Geofence radius containment check ($r \le r_{\text{landmark}}$).
  - Great-circle distance to nearest manned security post ($d_{\text{security}}$).
  - Campus zone lighting factor ($L \in [1.0, 10.0]$).
  - Spatial isolation index ($I \in [0.0, 1.0]$).
  - Diurnal schedule modifier (day vs. night risk penalties).
- **Model Nature:** **Deterministic Geospatial Engine**, NOT a trained machine learning model. No empirical crime or incident records exist for BIT Mesra to train a statistical classifier.

### Branch 3: Environmental Context Engine
- **Scientific Goal:** Quantify situational vulnerability based on diurnal cycles, lighting conditions, and surrounding pedestrian presence.
- **Implementation:** `src/environment/model.py`, `EnvironmentalContextEngine`.
- **Inputs:**
  - Time of day in Indian Standard Time (IST: UTC+5:30).
  - Ambient illumination (`Light`, `Dark`).
  - Crowd density (`Low`, `Medium`, `High`).
- **Mathematical Formulation:**
  $$P_3 = \text{clip}\left(W_{\text{time}}(h) + W_{\text{light}}(\text{light}) + W_{\text{crowd}}(\text{crowd}), 0.05, 1.0\right)$$
- **Model Nature:** **Rule-Based Contextual Evaluator**, NOT a trained statistical model.

---

## 3. Level 1 Stacking Ensemble Architecture

The stacking ensemble architecture fuses the three base outputs $\mathbf{X}_{\text{meta}} = [P_1, P_2, P_3]$ using a meta-learner:

$$\text{logit}(P_{\text{threat}}) = z = \beta_0 + \beta_1 P_1 + \beta_2 P_2 + \beta_3 P_3$$
$$P_{\text{threat}} = \frac{1}{1 + \exp(-z)}$$

### Production Separation vs. Simulation Isolation
To ensure absolute academic integrity and production safety:
1. **Production Pipeline (`src/fusion/predict.py`):**
   - Strictly refuses to load synthetic demo artifacts.
   - Enforces provenance validation at load time.
   - Reports `is_meta_learner_trained = False` and `meta_model_status = "UNTRAINED — VALID MULTIMODAL THREAT DATASET REQUIRED"`.
   - Executes the transparent **Experimental Weighted Context Index** (40% Physiological, 25% Spatial, 35% Environmental) labeled as a heuristic index.
2. **Synthetic Simulation Demonstration (`src/fusion/train.py` -> `run_synthetic_simulation_demo()`):**
   - Isolated demonstration workflow for validating mathematical mechanics, out-of-fold matrix generation, and calibration calculation.
   - Saves artifacts exclusively to `models/fusion/synthetic_demo/` and reports to `reports/synthetic_simulation/`.
   - All outputs are explicitly tagged `dataset_type = "synthetic_simulation"` and `production_eligible = False`.

---

## 4. Production Provenance Validation Gates

The system implements multiple strict validation gates in `src/fusion/stacking_pipeline.py`:

```python
class StackingEnsemblePipeline:
    @classmethod
    def load_artifact(cls, path, require_production_eligible=True):
        # 1. Path existence check
        # 2. Deserialization
        # 3. Provenance check:
        if require_production_eligible:
            if not is_eligible or dataset_type == "synthetic_simulation" or "synthetic" in str(path).lower():
                raise IneligibleProductionArtifactError(...)
        return pipeline
```

Furthermore, `StackingEnsemblePipeline.validate_multimodal_dataset()` enforces:
1. **Circularity Detection:** Evaluates whether target labels were generated by formulaic combinations of base features ($0.40 \cdot P_1 + 0.25 \cdot P_2 + 0.35 \cdot P_3$). Datasets with $\ge 90\%$ concordance are rejected with `DataCircularityError`.
2. **Synthetic Property Rejection:** When `enforce_production_empirical=True`, checks for synthetic subject tokens (`Subj_`, `Synthetic_`, `Sim_`) and raises `SyntheticDatasetRejectedError`.
3. **Class Distribution Check:** Requires at least two classes ($y \in \{0, 1\}$) with positive and negative instance support.

---

## 5. Software Component Mapping

| Component | File Path | Role | Status |
|---|---|---|---|
| Physiological ML Model | `src/physiological/model.py` | 76-feature WESAD stress classifier | Trained ML |
| Spatial Risk Engine | `src/location/model.py` | Deterministic Haversine campus analyzer | Deterministic |
| Campus Landmark Database | `config.py` | 17 verified BIT Mesra landmarks & security posts | Config / Data |
| Environmental Engine | `src/environment/model.py` | Rule-based situational evaluator | Rule-Based |
| Stacking Pipeline Logic | `src/fusion/stacking_pipeline.py` | Level 1 Stacking & Validation Gates | Implemented |
| Out-Of-Fold Generator | `src/fusion/oof_generator.py` | Leakage-free cross-validation matrix | Implemented |
| Training Workflows | `src/fusion/train.py` | Separated Production vs Demo training | Implemented |
| Unified Predictor | `src/fusion/predict.py` | Production inference engine | Untrained Meta-Learner |
| Streamlit Application | `app/main.py` | Interactive dashboard & GPS sensor | Active (Heuristic) |
| Synthetic Demo Model | `models/fusion/synthetic_demo/` | Mathematical demo artifact | Isolated Demo |
