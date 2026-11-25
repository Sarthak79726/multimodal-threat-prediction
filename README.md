# 🛡️ Multimodal Threat Prediction Engine
### Academic Personal Safety Analytics, Real-Data Ingestion & Dynamic Fusion Platform (BIT Mesra Campus)

[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.14-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/frontend-Streamlit%201.25+-FF4B4B.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/ML-scikit--learn-F7931E.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/ML-XGBoost-EB5424.svg)](https://xgboost.readthedocs.io/)
[![Open-Meteo](https://img.shields.io/badge/API-Open--Meteo%20Live%20Weather-008080.svg)](https://open-meteo.com/)
[![Tests](https://img.shields.io/badge/tests-104%20passed%20(100%25)-brightgreen.svg)](#-automated-testing--validation)
[![Meta-Learner Status](https://img.shields.io/badge/meta--learner-UNTRAINED-critical.svg)](#-important-academic--integrity-notice)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](#)

---

## 📌 Important Academic & Integrity Notice

> [!IMPORTANT]
> **Production Meta-Model Status:** `UNTRAINED — VALID MULTIMODAL THREAT DATASET REQUIRED`  
> **Active Production Inference:** `Modular Multi-Branch Fusion (Static Baseline + Dynamic Gating + Partial Dual-Branch)`  
> 
> A validated end-to-end multimodal threat-prediction meta-classifier cannot be trained until synchronized real-world multimodal threat observations and verified incident labels are collected.
> 
> - **Stress Detection $\neq$ Threat Prediction:** Biometric stress from WESAD reflects autonomic nervous system arousal (fight-or-flight), not an objective external threat.
> - **No Silent Neutral Substitution:** When a modality (e.g., environmental hardware) is missing or unavailable, the engine evaluates an explicit **Partial Dual-Branch Index** rather than fabricating synthetic data or assuming neutral values.
> - **Quarantine of Synthetic Benchmark Results:** All high-performance stacking metrics (ROC-AUC: 0.9939, Brier: 0.0181) originate from an **isolated synthetic simulation demonstration** used strictly to verify mathematical mechanics. They are quarantined in `reports/synthetic_simulation/` and `models/fusion/synthetic_demo/` and must never be cited as empirical threat-prediction results.

---

## 🏛️ System Architecture

The **Multimodal Threat Prediction Engine** is a B.Tech AIML research platform engineered for the **Birla Institute of Technology (BIT) Mesra** campus in Ranchi, Jharkhand, India (`23.4123°N, 85.4399°E`). It continuously synthesizes three foundational Level 0 data streams into actionable situational intelligence:

```
                                ┌────────────────────────────────────────────────────────┐
                                │          Live Client Telemetry / Web Browser           │
                                └───────────────────────────┬────────────────────────────┘
                                                            │
                      ┌─────────────────────────────────────┼─────────────────────────────────────┐
                      │                                     │                                     │
                      ▼                                     ▼                                     ▼
        ┌───────────────────────────┐         ┌───────────────────────────┐         ┌───────────────────────────┐
        │   1. Physiological Stream │         │    2. Location Stream     │         │   3. Environmental Stream │
        │ • WESAD Benchmark (S2–S17)│         │ • HTML5 Geolocation API   │         │ • Open-Meteo Live API     │
        │ • ECG, EDA, BVP, Resp, ACC│         │ • Haversine Geodesic      │         │   (Temp, Humidity, Rain)  │
        │ • 76 Biometric Features   │         │ • 17 BIT Mesra Landmarks  │         │ • Balanced Gradient Boost │
        │ • BLE Hardware Connector  │         │ • Ambiguity & Geofencing  │         │ • IoT Hardware Connectors │
        └─────────────┬─────────────┘         └─────────────┬─────────────┘         └─────────────┬─────────────┘
                      │                                     │                                     │
                      ▼                                     ▼                                     ▼
        ┌───────────────────────────┐         ┌───────────────────────────┐         ┌───────────────────────────┐
        │    Physiological Model    │         │    Spatial Risk Engine    │         │   Environmental Engine    │
        │   Random Forest Model     │         │   Deterministic Geodesic  │         │  Live API + Context AI    │
        │  [P_physio ∈ (0.0, 1.0)]  │         │  [P_spatial ∈ (0.0, 1.0)] │         │    [P_env ∈ (0.0, 1.0)]   │
        └─────────────┬─────────────┘         └─────────────┬─────────────┘         └─────────────┬─────────────┘
                      │                                     │                                     │
                      └──────────────────────────┐          │          ┌──────────────────────────┘
                                                 ▼          ▼          ▼
                                       ┌────────────────────────────────────────┐
                                       │   Real-Time Data Ingestion Hub         │
                                       │   • Cross-Modal Timestamp Sync         │
                                       │   • Sensor Health & Provenance Badges  │
                                       └───────────────────┬────────────────────┘
                                                           │
                                                           ▼
                                       ┌────────────────────────────────────────┐
                                       │     Dynamic Multimodal Fusion Engine   │
                                       │  1. Static Baseline: 0.40P+0.25L+0.35E │
                                       │  2. Dynamic Context-Sensitive Gating   │
                                       │  3. Partial Dual-Branch: (P+L) / 0.65  │
                                       │  4. Meta-Learner (Untrained Production)│
                                       └───────────────────┬────────────────────┘
                                                           │
                                                           ▼
                                       ┌────────────────────────────────────────┐
                                       │       Multimodal Risk Assessment       │
                                       │  • 5-Dimensional Risk Vector           │
                                       │  • Contextual Alert Level              │
                                       │  • Relative Modality Contributions     │
                                       └───────────────────┬────────────────────┘
                                                           │
                                                           ▼
                                       ┌────────────────────────────────────────┐
                                       │     Interactive Streamlit Dashboard    │
                                       │  • Tab 1: Threat Prediction Dashboard  │
                                       │  • Tab 2: GPS & Spatial Map Analyzer   │
                                       └────────────────────────────────────────┘
```

---

## 🔬 Modality Specifications & Provenance

| Modality Branch | Model / Engine | Data Source / Hardware Interface | Output Semantics | Operational Status |
|---|---|---|---|---|
| **1. Physiological** | `PhysiologicalStressModel`<br>*(Random Forest)* | **WESAD Benchmark** (`S2`–`S17`)<br>Live BLE Wearable Connector | $P(\text{Stress} \mid \mathbf{x}) \in [0, 1]$<br>Autonomic arousal | **Active (Trained ML)**<br>Hardware: `DISCONNECTED` |
| **2. Spatial** | `SpatialRiskEngine`<br>*(Geodesic Haversine)* | **HTML5 Browser Geolocation**<br>17 Surveyed BIT Mesra Landmarks | Exposure $\in [0, 1]$<br>Distance to Security (m) | **Active (Deterministic)**<br>Manual Override Available |
| **3. Environmental** | `EnvironmentalContextEngine`<br>*(Open-Meteo + XGBoost + RF)* | **Open-Meteo Live API** (`23.4123°N, 85.4399°E`)<br>Context Schedule + IoT Hardware Connectors | Vulnerability $\in [0, 1]$<br>Crowd, Lux, Weather | **Active (Live API + Model)**<br>IoT Sensors: `AWAITING_HARDWARE` |
| **4. Multimodal Fusion** | `DynamicMultimodalFusionEngine`<br>`MultimodalThreatPredictor` | Multi-branch late fusion with dynamic gating & missing-modality handling | Threat Score $\in [0, 1]$<br>5-Dimensional Risk Vector | **Active (Dynamic Late Fusion)** |
| **5. Meta-Learner** | `StackingEnsemblePipeline`<br>*(L2 Logistic Regression)* | Awaiting Synchronized Multimodal Ground Truth | Log-odds threat probability | **UNTRAINED (By Design)** |

---

## 🌐 Real-Time Data Ingestion Architecture (`src/ingestion/`)

The platform features an extensible real-time ingestion layer designed to ingest genuine sensor readings without ever generating synthetic replacement values:

1. **Live Open-Meteo Weather Client (`OpenMeteoWeatherClient`):**
   - Connects to the Open-Meteo Historical & Forecast Weather API for BIT Mesra coordinates (`lat=23.4123, lon=85.4399`).
   - Retrieves authentic ambient temperature (°C), relative humidity (%), precipitation (mm), cloud cover (%), and daylight status.
   - Built with thread-safe singleton patterns, a 15-minute TTL memory cache, and strict physical range validation.
   - Operates with zero synthetic fabrication: if network connectivity is interrupted, it reports `UNAVAILABLE` without crashing or inventing fake weather.

2. **Physical Lighting Sensor Connector (`PhysicalLightingSensorConnector`):**
   - Provides standardized hardware hooks for physical photometers and I2C/SPI lux sensors (e.g., BH1750, TSL2561).
   - Reports physical lux measurements or flags status as `AWAITING_HARDWARE` when physical equipment is not attached.

3. **Privacy-Preserving Crowd Occupancy Connector (`PrivacyPreservingCrowdConnector`):**
   - Ingests aggregated occupant counts from privacy-preserving hardware (PIR arrays, thermal tripwires, turnstile counters).
   - Validates non-negative counts and reports status as `AWAITING_HARDWARE` in laboratory/demonstration mode.

4. **Wearable BLE Streaming Connector (`WearableDeviceConnector`):**
   - Ingests live continuous telemetry packets (BVP, EDA, TEMP, ECG, ACC) from Bluetooth Low Energy (BLE) wearables.
   - Flags connection state as `DISCONNECTED` or replays authentic research sessions from the WESAD benchmark.

5. **Real-Time Data Ingestion Hub (`RealDataIngestionHub`):**
   - Coordinates multi-sensor streaming, cross-modal timestamp alignment within strict temporal tolerance windows, and unified provenance metadata generation.

---

## ⚖️ Multimodal Fusion Engine & Risk Vector

The fusion architecture provides four distinct evaluation modes:

### 1. Static Late Fusion Baseline
A transparent, linear weighted combination calibrated for campus personal safety:
$$R_{\text{static}} = 0.40 \cdot P_{\text{physio}} + 0.25 \cdot P_{\text{spatial}} + 0.35 \cdot P_{\text{env}}$$

### 2. Dynamic Context-Sensitive Gating
Adapts modality contributions dynamically based on situational context and sensor availability:
- **Nighttime / Poor Lighting Gating:** When illumination drops below critical thresholds (< 5 lux), environmental vulnerability weight automatically scales up.
- **High Spatial Isolation Gating:** When distance to the nearest guarded security post exceeds 300 meters, spatial weight expands proportionally.
- **Biometric Confidence Gating:** If wearable signal-to-noise ratio is degraded or resting state is detected, physiological weight is safely throttled.

### 3. Partial Dual-Branch Context Index (Missing Modality Handling)
When environmental context is unavailable (e.g., hardware disconnected or outside campus boundary), the engine evaluates a normalized dual-branch index:
$$R_{\text{dual}} = \frac{0.40 \cdot P_{\text{physio}} + 0.25 \cdot P_{\text{spatial}}}{0.40 + 0.25} = \frac{0.40 \cdot P_{\text{physio}} + 0.25 \cdot P_{\text{spatial}}}{0.65}$$
*Crucially, no synthetic neutral values (such as 0.20 or 0.50) are silently substituted.*

### 4. Multimodal Risk Vector (5 Dimensions)
Rather than collapsing all situational data into a single opaque number, the engine produces an explicit 5-dimensional risk vector:
1. **Physiological Distress Assessment:** Wearable autonomic arousal index.
2. **Spatial Exposure Assessment:** Distance to safety, geofencing, and isolation.
3. **Environmental Risk Assessment:** Schedule-based density, lux, and live weather.
4. **Contextual Situation Index:** Composite ambient vulnerability index.
5. **Overall Multimodal Threat Level:** Unified situational context classification (`LOW`, `MODERATE`, `ELEVATED`, `HIGH`).

---

## 🗂️ Project Directory Layout

```
project1/
├── app/
│   ├── main.py                     # Interactive Streamlit application (Tabs 1 & 2)
│   └── styles.css                  # Modern UI stylesheet with provenance badges
├── config.py                       # BIT Mesra campus landmarks, coordinates & tolerances
├── src/
│   ├── ingestion/                  # Real-time sensor & API ingestion platform
│   │   ├── schemas.py              # Strongly-typed data schemas and enums
│   │   ├── weather_client.py       # Live Open-Meteo Weather API client (BIT Mesra)
│   │   ├── connectors.py           # Physical Lux, Crowd, Wearable & GPS connectors
│   │   └── data_synchronizer.py    # Cross-modal timestamp sync & RealDataIngestionHub
│   ├── fusion/                     # Multimodal fusion architecture
│   │   ├── dynamic_engine.py       # Dynamic context gating & 5D risk vector
│   │   ├── meta_learner.py         # Level 1 Stacking Ensemble pipeline
│   │   ├── predict.py              # MultimodalThreatPredictor production facade
│   │   └── train.py                # Stacking training pipeline (quarantined demo)
│   ├── physiological/              # Level 0 Biometric stress branch
│   │   ├── model.py                # PhysiologicalStressModel inference engine
│   │   ├── feature_extraction.py   # 76-feature extraction (ECG, EDA, Resp, Temp, ACC)
│   │   ├── loader.py               # WESAD subject data loader (S2–S17)
│   │   └── train.py                # Random Forest training pipeline
│   ├── location/                   # Level 0 Geospatial branch
│   │   ├── campus_map.py           # CampusMapAnalyzer & Haversine distance engine
│   │   └── risk_scorer.py          # Spatial risk evaluator & security proximity
│   └── environment/                # Level 0 Environmental branch
│       ├── model.py                # EnvironmentalContextEngine (Crowd + Lighting AI)
│       └── train.py                # Gradient Boosting (Crowd) & Random Forest (Lux)
├── models/                         # Serialized model artifacts & metadata
│   ├── physiological/              # physiological_model.pkl (Random Forest)
│   ├── location/                   # location_model.pkl
│   ├── environment/                # crowd_model.pkl, lighting_model.pkl
│   └── fusion/                     # stacking_ensemble.pkl (untrained in prod)
├── docs/                           # Comprehensive academic documentation suite
│   ├── TECHNICAL_ARCHITECTURE.md   # System decomposition and runtime flow
│   ├── STACKING_METHODOLOGY.md     # OOF cross-validation and mathematical proofs
│   ├── DATASET_AND_LABEL_REQUIREMENTS.md # Data collection and ethical ground truth
│   ├── MODEL_EVALUATION.md         # Calibration curves, Brier score, ECE metrics
│   └── SCIENTIFIC_LIMITATIONS.md   # Candid academic disclosures and ethics
└── tests/                          # 104 automated unit, integration & regression tests
    ├── test_real_data_platform.py  # Ingestion hub, Open-Meteo API & hardware tests
    ├── test_dynamic_fusion.py      # Dynamic gating & missing modality unit tests
    ├── test_wesad_subject_switching.py # Subject & condition isolation tests (S2–S17)
    ├── test_audit_v1_corrections.py # Geofencing, ambiguity & regression tests
    └── ... (12 test modules total)
```

---

## 🧪 Isolated Synthetic Simulation Demonstration

To verify that the mathematical mechanics of Out-of-Fold (OOF) cross-validation and Level 1 Stacking work correctly without numerical instability, a synthetic simulation was executed across 4,000 synthetic observations with 20 artificial subject tokens (`Subj_01` to `Subj_20`).

The results are preserved strictly for algorithmic verification:
- **Simulation Artifact:** `models/fusion/synthetic_demo/stacking_ensemble_demo.pkl`
- **Benchmark Report:** `reports/synthetic_simulation/stacking_benchmark_results_synthetic_demo.json`
- **Evaluation Plots:** `reports/synthetic_simulation/`

> [!NOTE]
> The demonstration confirmed that the stacking formulation achieves over 9× lower calibration error (Brier Score: 0.0181 vs. 0.1086; ECE: 0.0301 vs. 0.2718) than a raw weighted sum on non-linear distributions. However, because these inputs were generated mathematically, they demonstrate **simulation mechanics only** and are not empirical threat predictions.

---

## 🚀 Quickstart Guide

### 1. Environment Setup
```bash
# Clone the repository
git clone https://github.com/Sarthak79726/multimodal-threat-prediction.git
cd multimodal-threat-prediction

# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1       # Windows PowerShell
# or: source .venv/bin/activate  # Linux / macOS

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Comprehensive Test Suite
Execute the entire test suite (104 automated tests covering ingestion, fusion, models, and UI):
```bash
python -m unittest discover tests -v
```

### 3. Launch Streamlit Application
```bash
streamlit run app/main.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🌐 Streamlit Cloud Deployment Guide

When deploying to [Streamlit Community Cloud](https://share.streamlit.io):
1. **Repository:** Connect your GitHub repository (`multimodal-threat-prediction`).
2. **Branch:** `main`
3. **Main file path:** `app/main.py`
4. **Python Version:** 3.10+ (recommended: 3.11)
5. **Rebooting / Clearing Cache:** If updating models or dependencies, click **Manage app** in the lower-right corner $\to$ **⋮** (three dots) $\to$ **Reboot app** or **Clear cache and reboot** to pull the latest commit container.

---

## 🔬 Automated Testing & Validation

The test suite contains **104 automated tests** with 100% pass rate:
- `test_real_data_platform.py`: Validates Open-Meteo live API integration, caching, range validation, physical lux/crowd connectors, BLE wearable connector, and `RealDataIngestionHub`.
- `test_dynamic_fusion.py`: Verifies dynamic context-sensitive gating, 5D risk vector, and Partial Dual-Branch fallback when environmental data is missing.
- `test_wesad_subject_switching.py`: Audits subject selection across all 15 usable WESAD participants (`S2`–`S17`) and verifies condition isolation (Baseline, Stress, Amusement, Meditation).
- `test_audit_v1_corrections.py`: Verifies GPS geofencing, ambiguity detection, and ensures `resolve_campus_location` never raises a `TypeError`.
- `test_location_override.py`: Confirms manual landmark override replaces GPS detection for current inference without breaking underlying GPS tracking.

```bash
# Run all 104 tests
python -m unittest discover tests -v
```

---

## 📄 Academic Integrity & License

This project was conceived and developed as an academic research project at the **Birla Institute of Technology, Mesra**. Released under the [MIT License](LICENSE).
