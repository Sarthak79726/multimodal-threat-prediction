# 🛡️ Multimodal Threat Prediction Engine
### Academic Personal Safety Analytics & Stacking Ensemble Architecture (BIT Mesra Campus)

[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.14-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/frontend-Streamlit%201.25+-FF4B4B.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/ML-scikit--learn-F7931E.svg)](https://scikit-learn.org/)
[![Status](https://img.shields.io/badge/meta--learner-UNTRAINED-critical.svg)](#)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](#)

---

## 📌 Important Academic & Integrity Notice

> [!IMPORTANT]
> **Production Meta-Model Status:** `UNTRAINED — VALID MULTIMODAL THREAT DATASET REQUIRED`  
> **Active Inference Score:** `Experimental Weighted Context Index`  
> 
> A validated multimodal threat-prediction model is not currently available because synchronized multimodal observations and independently verified real-world outcome labels have not yet been collected.
> 
> **This is a heuristic index, not a calibrated probability of physical threat.**
> 
> All high-performance stacking metrics previously reported (ROC-AUC: 0.9939, Brier: 0.0181) originate from an **isolated synthetic simulation demonstration** used exclusively to verify mathematical pipeline mechanics. They are strictly quarantined in `reports/synthetic_simulation/` and `models/fusion/synthetic_demo/` and must not be cited as empirical threat-prediction results.

---

## 🏛️ System Architecture

The **Multimodal Threat Prediction Engine** is a B.Tech AIML research project developed for the **Birla Institute of Technology (BIT) Mesra** campus in Ranchi, India. It investigates the integration of three distinct analytical streams:

1. **Level 0 — Physiological Stress Branch:** Continuous biometric classification ($P_{\text{physio}} = P(\text{Stress} \mid \mathbf{x})$) derived from 76 clinical features across chest and wrist sensors from the WESAD benchmark dataset. *(Trained ML: Random Forest)*.
2. **Level 0 — Geospatial Campus Branch:** Geofencing and great-circle Haversine distance engine mapping live browser GPS coordinates to 17 verified BIT Mesra campus landmarks, guarded security checkpoints, spatial isolation indices, and zone lighting. *(Deterministic Geodesic Engine)*.
3. **Level 0 — Environmental Context Branch:** Dynamic situational vulnerability assessment reflecting campus schedule (IST: UTC+5:30), ambient illumination, and surrounding crowd density. *(Rule-Based Context Evaluator)*.
4. **Level 1 — Stacking Meta-Learner (Architecture Implemented):** An L2-regularized Logistic Regression meta-classifier designed to learn associative log-odds coefficients ($\beta_0, \beta_1, \beta_2, \beta_3$) via Out-of-Fold (OOF) cross-validation. Currently **untrained** awaiting real-world multimodal threat data.
5. **Active Production Baseline:** The system operates using the **Experimental Weighted Context Index** ($0.40 \cdot P_{\text{physio}} + 0.25 \cdot P_{\text{spatial}} + 0.35 \cdot P_{\text{env}}$) with explicit heuristic labeling and context indicators (`STANDARD CONTEXT` / `ELEVATED CONTEXT RISK`).

```
                               ┌──────────────────────────────────────────────┐
                               │     Live Client Telemetry / Web Browser      │
                               └──────────────────────┬───────────────────────┘
                                                      │
                    ┌─────────────────────────────────┼─────────────────────────────────┐
                    │                                 │                                 │
                    ▼                                 ▼                                 ▼
        [ 1. Physiological Stream ]        [ 2. Location Stream ]            [ 3. Environment Stream ]
        • WESAD Chest / Wrist Signals      • HTML5 Browser Geolocation       • BIT Mesra Time (IST)
        • ECG, EDA, BVP, Temp, ACC         • Haversine Distance Engine       • Illumination (Light / Dark)
        • 76 Extracted Features            • 17 BIT Mesra Campus Landmarks   • Crowd Density Level
                    │                                 │                                 │
                    ▼                                 ▼                                 ▼
        ┌────────────────────────┐        ┌────────────────────────┐        ┌────────────────────────┐
        │   Physiological Model  │        │   Spatial Risk Engine  │        │  Environmental Engine  │
        │ Trained Random Forest  │        │ Deterministic Geodesic │        │ Contextual Rule-Based  │
        │ [P_physio ∈ (0.0, 1.0)]│        │ [P_spatial ∈(0.0, 1.0)]│        │  [P_env ∈ (0.0, 1.0)]  │
        └───────────┬────────────┘        └───────────┬────────────┘        └───────────┬────────────┘
                    │                                 │                                 │
                    └────────────────────────┐        │        ┌────────────────────────┘
                                             ▼        ▼        ▼
                                   ┌────────────────────────────────────┐
                                   │  Meta-Feature Vector X_meta ∈ R^3  │
                                   │  [P_physio, P_spatial, P_env]      │
                                   └─────────────────┬──────────────────┘
                                                     │
                                                     ▼
                                   ┌────────────────────────────────────┐
                                   │ PRODUCTION DECISION:               │
                                   │ Meta-Learner: UNTRAINED            │
                                   │ Active: Experimental Weighted      │
                                   │ Context Index (Heuristic 40/25/35) │
                                   └─────────────────┬──────────────────┘
                                                     │
                                                     ▼
                                   ┌────────────────────────────────────┐
                                   │    Interactive Streamlit UI        │
                                   │    • Status: Meta-Learner Untrained│
                                   │    • Transparent Modality Badges   │
                                   │    • OpenStreetMap Spatial Layer   │
                                   └────────────────────────────────────┘
```

---

## 📚 Comprehensive Documentation Suite

Detailed architectural, mathematical, and ethical specifications are maintained in the [`docs/`](docs/) directory:

- [**Technical Architecture (`docs/TECHNICAL_ARCHITECTURE.md`)**](docs/TECHNICAL_ARCHITECTURE.md): System decomposition, component layout, and runtime inference flow.
- [**Stacking Methodology (`docs/STACKING_METHODOLOGY.md`)**](docs/STACKING_METHODOLOGY.md): Mathematical formulation, Out-Of-Fold (OOF) cross-validation protocol, and separation of simulation from production.
- [**Dataset & Label Requirements (`docs/DATASET_AND_LABEL_REQUIREMENTS.md`)**](docs/DATASET_AND_LABEL_REQUIREMENTS.md): Data collection specifications, sensor synchronization tolerances, subject diversity, and ground truth label requirements.
- [**Model Evaluation (`docs/MODEL_EVALUATION.md`)**](docs/MODEL_EVALUATION.md): Evaluation metrics (Brier, ECE, PR-AUC), synthetic simulation benchmarks, and the production deployment checklist.
- [**Scientific Limitations (`docs/SCIENTIFIC_LIMITATIONS.md`)**](docs/SCIENTIFIC_LIMITATIONS.md): Candid disclosure of data limitations, physiological stress vs. threat semantics, and ethical life-safety warnings.

---

## 🔬 Modality Provenance & Semantics

| Modality Branch | Model Implementation | Dataset / Source | Output Semantics | Classification Nature |
|---|---|---|---|---|
| **1. Physiological** | `PhysiologicalStressModel` | WESAD (Wearable Stress Benchmark) | $P(\text{Stress} \mid \mathbf{x}) \in [0, 1]$ | **Trained ML** (Random Forest) |
| **2. Spatial** | `SpatialRiskEngine` | BIT Mesra Campus Landmarks & Geodesics | Geodesic vulnerability $\in [0, 1]$ | **Deterministic Engine** (Haversine) |
| **3. Environmental**| `EnvironmentalContextEngine` | BIT Mesra Synthetic Environmental Dataset (73,440 samples) | Contextual vulnerability $\in [0, 1]$, Crowd Category, Lighting Category, Lux Estimate | **Trained ML (Experimental)** (XGBoost & Random Forest) |
| **4. Meta-Learner** | `StackingEnsemblePipeline` | Awaiting Empirical Multimodal Data | — | **UNTRAINED** |
| **5. Active Index** | `MultimodalThreatPredictor` | Heuristic Linear Combination | Context Index $\in [0, 1]$ | **Experimental Heuristic Index** |

---

## 🧪 Isolated Synthetic Simulation Demonstration

To verify that the mathematical mechanics of Out-of-Fold training and Stacking work correctly without numeric instability, a synthetic simulation demonstration was executed on 4,000 synthetic observations with 20 artificial subject tokens (`Subj_01` to `Subj_20`).

The results are preserved strictly for mathematical reproducibility in:
- Demo Artifact: `models/fusion/synthetic_demo/stacking_ensemble_demo.pkl`
- Benchmark Report: `reports/synthetic_simulation/stacking_benchmark_results_synthetic_demo.json`
- Plots: `reports/synthetic_simulation/`

> [!NOTE]
> The synthetic demonstration confirmed that the stacking formulation achieves over 9× lower calibration error (Brier Score: 0.0181 vs. 0.1086; ECE: 0.0301 vs. 0.2718) than a raw weighted sum on non-linear distributions. However, because the input distributions were generated using NumPy Beta distributions, these metrics represent **simulation mechanics only**.

---

## 🚀 Quickstart & Reproduction Guide

### 1. Environment Setup
```bash
# Clone the repository
git clone https://github.com/Sarthak79726/multimodal-threat-prediction.git
cd multimodal-threat-prediction

# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1   # Windows PowerShell
# or: source .venv/bin/activate  (Linux/macOS)

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Comprehensive Unit & Regression Tests
```bash
python -m unittest discover tests -v
```

### 3. Launch Interactive Streamlit Dashboard
```bash
streamlit run app/main.py
```

### 4. Execute Environmental AI & Meta-Learner Workflows
```bash
# A. Generate 73,440-record Synthetic Environmental Dataset:
python src/environment/generate_synthetic_dataset.py

# B. Train Environmental Context AI Models (Model A: Crowd, Model B: Lighting):
python src/environment/train.py

# C. Run Isolated Synthetic Simulation Demo (Outputs to models/fusion/synthetic_demo/):
python src/fusion/train.py --demo

# D. Attempt Production Training (Strictly requires real empirical dataset):
python src/fusion/train.py --production --data-path path/to/real_multimodal_data.csv
```
*(If no empirical dataset is provided, production training safely terminates with a descriptive error and leaves the meta-model untrained).*

---

## 📄 License & Academic Integrity

This project is released under the MIT License. Developed for academic research purposes at BIT Mesra.
