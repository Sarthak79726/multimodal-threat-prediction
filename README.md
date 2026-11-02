# 🛡️ Multimodal Threat Prediction Engine
### Spatial, Physiological & Environmental Risk Assessment System (BIT Mesra Campus)

[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.14-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/frontend-Streamlit%201.25+-FF4B4B.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/ML-scikit--learn-F7931E.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/ML-XGBoost-1199EE.svg)](https://xgboost.readthedocs.io/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](#)

---

## 📌 Executive Overview

The **Multimodal Threat Prediction Engine** is a real-time, explainable personal safety and threat-risk estimation platform. Developed with tailored geospatial telemetry for the **Birla Institute of Technology (BIT) Mesra** campus in Ranchi, India, the system fuses three distinct data streams to compute a holistic, calibrated threat probability:

1. **Wearable Physiological Stress Signals:** Continuous biometric analysis (EDA, ECG, BVP, Respiration, Accelerometer) derived from the clinical WESAD benchmark dataset.
2. **Geospatial Campus Geofencing & Location Telemetry:** Live browser GPS geolocation mapped against BIT Mesra campus landmarks, isolation indices, lighting ratings, and distance to 24/7 security posts.
3. **Environmental & Situational Context:** Dynamic parameters including localized campus time of day (IST: UTC+5:30), ambient illumination, and surrounding crowd density.

The predictions are integrated via an explainable meta-learner and weighted decision fusion pipeline, delivering actionable risk classifications (`SAFE` vs. `RISKY`) alongside natural language diagnostic rationales.

---

## 🏛️ System Architecture & Multimodal Pipeline

```
                                ┌──────────────────────────────────────────────┐
                                │          Live Client / Web Browser           │
                                └──────────────────────┬───────────────────────┘
                                                       │
                     ┌─────────────────────────────────┼─────────────────────────────────┐
                     │                                 │                                 │
                     ▼                                 ▼                                 ▼
         [ 1. Physiological Stream ]        [ 2. Location Stream ]            [ 3. Environment Stream ]
         • WESAD Chest / Wrist Signals      • HTML5 Browser Geolocation       • BIT Mesra Time (IST)
         • ECG, EDA, BVP, Temp, ACC         • Haversine Distance Engine       • Lighting (Light / Dark)
         • GroupKFold Cross-Validation      • Campus Landmarks Database       • Crowd Density Level
                     │                                 │                                 │
                     ▼                                 ▼                                 ▼
         ┌────────────────────────┐        ┌────────────────────────┐        ┌────────────────────────┐
         │  Physiological Model   │        │     Location Model     │        │   Environment Model    │
         │  (RandomForest/XGBoost)│        │ (GradientBoost/XGBoost)│        │  (Calibrated Baseline) │
         └───────────┬────────────┘        └───────────┬────────────┘        └───────────┬────────────┘
                     │                                 │                                 │
                     │ Score (40% Weight)              │ Score (25% Weight)              │ Score (35% Weight)
                     └────────────────────────┐        │        ┌────────────────────────┘
                                              ▼        ▼        ▼
                                    ┌────────────────────────────────────┐
                                    │      Multimodal Fusion Engine      │
                                    │    • Weighted Prototype Model      │
                                    │    • Meta-Learner Classification   │
                                    │    • Automated Rationale Generator │
                                    └─────────────────┬──────────────────┘
                                                      │
                                                      ▼
                                    ┌────────────────────────────────────┐
                                    │    Interactive Streamlit UI        │
                                    │    • Clinical Analytics Theme      │
                                    │    • OpenStreetMap Spatial Layer   │
                                    │    • Mobile Hardened Controls      │
                                    └────────────────────────────────────┘
```

---

## 🛠️ Complete Tech Stack Breakdown

### 1. Machine Learning & Signal Processing
| Technology | Version / Tool | Purpose & Application |
|---|---|---|
| **Python** | `3.10+` / `3.11` / `3.14` | Primary runtime language for ML pipelines, APIs, and the interactive app. |
| **scikit-learn** | `>=1.2.0` | GroupKFold / StratifiedKFold validation, preprocessing (`StandardScaler`, `OneHotEncoder`, `ColumnTransformer`), baseline classifiers (`LogisticRegression`, `RandomForestClassifier`, `GradientBoostingClassifier`, `MultinomialNB`), and evaluation metrics (`ROC-AUC`, `F1-Score`, `Precision`, `Recall`). |
| **XGBoost** | `>=1.7.0` | Gradient-boosted decision trees for tabular physiological, environmental, and spatial feature classification. |
| **SciPy** | `>=1.10.0` | Peak detection (`scipy.signal.find_peaks`) for ECG/BVP inter-beat interval extraction, heart rate (BPM) calculation, and statistical moment evaluation (`scipy.stats.skew`). |
| **Joblib & Pickle** | Standard / `>=1.2.0` | Model artifact persistence and serialization across `models/` directories. |

### 2. Data Science & Spatial Analytics
| Technology | Version / Tool | Purpose & Application |
|---|---|---|
| **NumPy** | `>=1.24.0` | Multi-dimensional numerical array manipulations, signal window slicing, vector norms, and Haversine distance computations. |
| **Pandas** | `>=2.0.0` | Tabular data processing, dataset synthesis, feature matrix formatting, and geographic landmark DataFrame handling. |
| **Geodesic Engine** | Custom Haversine | Great-circle distance calculations between user device coordinates and BIT Mesra campus landmarks. |

### 3. Frontend & Interactive Dashboard
| Technology | Version / Tool | Purpose & Application |
|---|---|---|
| **Streamlit** | `>=1.25.0` | Core application framework providing reactive state management (`st.session_state`), tab navigation, and responsive layouts. |
| **Streamlit Custom Components** | `streamlit.components.v1` | Bi-directional integration bridge loading custom HTML5/JavaScript sensor widgets. |
| **streamlit-js-eval** | `>=0.1.5` | Client-side JavaScript execution helper inside the Streamlit session. |
| **Plotly Express & Graph Objects** | Optional / Auto-detected | Renders OpenStreetMap interactive geospatial maps (`px.scatter_map`) with landmark badges and live GPS positions. |
| **Matplotlib & Seaborn** | `>=3.7.0` / `>=0.12.0` | Offline waveform visualizations, exploratory data analysis, and distribution plotting in `notebooks/`. |

### 4. Custom Hardware Sensor & UI Layer (Web/CSS/JS)
| Technology | File / Module | Purpose & Application |
|---|---|---|
| **HTML5 Geolocation API** | `app/components/gps/index.html` | High-accuracy device GPS tracking (`watchPosition`, `getCurrentPosition`) with caching and automatic fallback. |
| **Custom Clinical CSS** | `app/static/dashboard.css` | Medical-grade styling, custom color variables, score badges, status pills, and card elevation. |
| **Decoupled HTML Templates** | `app/templates/dashboard.html` | Presentation templates for the dashboard header and overall risk breakdown card. |
| **Mobile Hardening Script** | Embedded in `app/main.py` | MutationObserver script enforcing `readOnly` attributes on select inputs to prevent keyboard popups on mobile devices. |

### 5. Infrastructure, Containers & DevOps
| Technology | File / Tool | Purpose & Application |
|---|---|---|
| **VS Code Dev Containers** | `.devcontainer/devcontainer.json` | Reproducible development container based on `python:1-3.11-bookworm` with auto-port forwarding on `8501`. |
| **Git & GitHub** | `main` branch | Version control repository (`multimodal-threat-prediction`). |

---

## 📂 Project Directory Structure

```plaintext
multimodal-threat-prediction/
│
├── .devcontainer/                  # VS Code Codespaces / Dev Container config
│   └── devcontainer.json
├── app/                            # Interactive Dashboard Application
│   ├── components/                 # Custom Bi-directional Streamlit Components
│   │   └── gps/
│   │       └── index.html          # HTML5 Geolocation sensor component
│   ├── static/
│   │   └── dashboard.css           # Medical analytics stylesheet
│   ├── templates/
│   │   └── dashboard.html          # HTML presentation templates
│   └── main.py                     # Streamlit application entry point
│
├── config.py                       # Global configuration, landmarks, paths & IST timezone
├── requirements.txt                # Production Python dependencies
│
├── data/                           # Data storage directory
│   ├── WESAD/                      # Raw WESAD dataset (subjects S2–S17) & readme
│   ├── physiological/              # Extracted physiological feature datasets
│   ├── location/                   # Synthesized campus location datasets
│   ├── environment/                # Synthesized environmental factor datasets
│   ├── content/                    # Synthesized text content datasets
│   ├── context/                    # Synthesized context datasets
│   └── fusion/                     # Multimodal fusion datasets
│
├── models/                         # Trained model artifacts (.pkl files)
│   ├── physiological/              # physiological_model.pkl
│   ├── location/                   # location_model.pkl
│   ├── environment/                # environment_model.pkl
│   ├── fusion/                     # fusion_model.pkl
│   ├── content/                    # content_model.pkl
│   └── context/                    # context_model.pkl
│
├── notebooks/                      # Exploratory Data Analysis & Notebooks
│   ├── 01_wesad_exploration.ipynb
│   ├── explore_wesad.py            # WESAD inspection script & waveform plotter
│   └── plots/                      # Generated visualization waveforms
│
└── src/                            # Core Source Code & ML Engine
    ├── content/                    # Text/content NLP classification branch
    │   ├── generate_data.py
    │   └── train.py
    ├── context/                    # Context analysis branch
    │   ├── generate_data.py
    │   └── train.py
    ├── environment/                # Environmental risk modeling branch
    │   ├── generate_data.py
    │   └── train.py
    ├── location/                   # Geospatial campus analysis & models
    │   ├── campus_map.py           # Haversine distance & spatial metrics analyzer
    │   ├── landmarks.py            # Centralized BIT Mesra landmarks database
    │   ├── generate_data.py
    │   └── train.py
    ├── physiological/              # WESAD preprocessing & feature extraction
    │   ├── preprocessing.py        # Subject loader & signal inspector
    │   ├── feature_extraction.py   # Signal statistics, HRV, RMSSD, ECG/BVP peaks
    │   ├── inspect_wesad.py
    │   ├── visualize_signals.py
    │   └── train.py                # GroupKFold training across subjects
    └── fusion/                     # Multimodal Decision Fusion
        ├── fusion_dataset.py       # Fusion dataset generator
        ├── train.py                # Stacking Meta-Learner training
        └── predict.py              # Unified MultimodalThreatPredictor inference engine
```

---

## 📊 Modality Branches Explained

### 1. Physiological Biometric Branch
- **Dataset:** WESAD (*Wearable Stress and Affect Detection*), containing chest and wrist recordings across 15 participants.
- **Chest Modalities (700 Hz):** ECG, EDA, EMG, Respiration, Skin Temperature, 3-axis Accelerometer.
- **Wrist Modalities:** BVP (64 Hz), EDA (4 Hz), Skin Temperature (4 Hz), 3-axis Accelerometer (32 Hz).
- **Engineered Features:**
  - Statistical moments: Mean, Standard Deviation, Min, Max, Skewness, Linear Slope Trend.
  - Heart Rate Variability (HRV): Inter-beat intervals, Mean RR interval, RR standard deviation, RMSSD, BPM.
  - Accelerometer vector magnitude ($|ACC| = \sqrt{x^2 + y^2 + z^2}$).

### 2. Geospatial Campus Branch (BIT Mesra)
- Centralized database of coordinates across campus zones:
  - **Administrative Hub:** Clock Tower Main Building (active security, CCTV, high lighting).
  - **Academic Zones:** R&D Building, Central Library, Department complexes.
  - **Residential Hostels:** Hostels 1 through 14.
  - **Isolated Sectors:** Ring Road perimeters, sports grounds, forest fringes.
- Calculates dynamic spatial metrics:
  - Distance to nearest landmark & detection radius boundary check.
  - Distance to active 24/7 campus security posts.
  - Environmental isolation index ($0.0 - 1.0$) and time-adjusted zone illumination ($1.0 - 10.0$).

### 3. Environmental Context Branch
- Evaluates ambient conditions affecting situational vulnerability:
  - **Campus Local Time (IST):** Differentiates between daytime, twilight/evening, and late night ($23:00 - 04:30$).
  - **Lighting Conditions:** Light vs. Dark illumination.
  - **Crowd Density:** Low (isolated), Medium (normal), High (populated public area).

### 4. Multimodal Decision Fusion Engine
The inference engine (`src/fusion/predict.py`) applies a transparent, clinically calibrated decision weighting:
$$\text{Threat Score (\%)} = \left(0.40 \times S_{\text{physio}} + 0.25 \times S_{\text{location}} + 0.35 \times S_{\text{environment}}\right) \times 100$$
- **Classification Threshold:** $\ge 50.0\%$ triggers `RISKY`, otherwise `SAFE`.
- **Confidence Tiers:**
  - `LOW`: $\le 39\%$
  - `MEDIUM`: $40\% - 69\%$
  - `HIGH`: $\ge 70\%$
- **Diagnostic Explainability:** Automatically generates diagnostic text explaining which specific branch elevated the score (e.g., physiological surge vs. late-night secluded location).

---

## 🚀 Quickstart & Setup Guide

### 1. Clone the Repository
```bash
git clone https://github.com/Sarthak79726/multimodal-threat-prediction.git
cd multimodal-threat-prediction
```

### 2. Create and Activate a Virtual Environment
```bash
# On Linux / macOS:
python3 -m venv .venv
source .venv/bin/activate

# On Windows (PowerShell):
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```
*(Optional for enhanced mapping visualization)*:
```bash
pip install plotly
```

### 4. Run the Streamlit Application
```bash
streamlit run app/main.py
```
Open your browser at `http://localhost:8501`.

---

## 🧪 Retraining Models & Generating Datasets

To retrain any of the modality branches or regenerate synthetic datasets:

```bash
# 1. Physiological Model (uses extracted WESAD features)
python src/physiological/train.py

# 2. Location Risk Model
python src/location/train.py

# 3. Environment Risk Model
python src/environment/train.py

# 4. Multimodal Fusion Model
python src/fusion/train.py
```

---

## 📱 Mobile Compatibility & Geolocation Permissions

- **Live Browser GPS:** Works on mobile and desktop browsers over HTTPS or `localhost`.
- **Android / iOS Hardening:** Includes a custom script preventing virtual keyboards from popping up when interacting with dropdowns and cards.
- **Location Caching:** Automatically caches recent GPS fixes to ensure fluid dashboard performance during rapid navigation.

---

## 📄 License
This project is developed for academic research, threat prediction benchmarking, and campus safety optimization. Licensed under the MIT License.
