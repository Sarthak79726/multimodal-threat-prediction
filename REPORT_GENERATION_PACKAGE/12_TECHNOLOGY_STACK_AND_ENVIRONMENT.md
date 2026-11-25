# Multimodal Threat Prediction Engine — Version 1.0
## Technical Documentation: Technology Stack, Runtime Environment, and Dependency Specifications

---

### 1. Primary Technology Stack Summary

The Multimodal Threat Prediction Engine is built entirely on modern open-source scientific Python libraries, utilizing standard machine learning frameworks, geospatial spherical trigonometry, and an asynchronous web dashboard.

```text
┌────────────────────────────────────────────────────────────────────────┐
│                        PRESENTATION & UI LAYER                         │
│  Streamlit (1.62.0)  •  Folium (Leaflet.js)  •  HTML5 Geolocation API  │
├────────────────────────────────────────────────────────────────────────┤
│                       SERVICE & INFERENCE LAYER                        │
│     MultimodalThreatPredictor  •  DynamicMultimodalFusionEngine        │
│          Dynamic Gating (Softmax)  •  Missing-Data Renormalizer         │
├────────────────────────────────────────────────────────────────────────┤
│                    MACHINE LEARNING & MODELING LAYER                   │
│   Scikit-Learn (1.9.0)  •  XGBoost (3.4.1)  •  Joblib & Pickle Serial  │
├────────────────────────────────────────────────────────────────────────┤
│                      SCIENTIFIC COMPUTING & DATA                       │
│    NumPy (2.5.2)  •  Pandas (3.0.5)  •  SciPy Signal & Stats (1.18.1)  │
├────────────────────────────────────────────────────────────────────────┤
│                        SYSTEM & RUNTIME RUNTIME                        │
│   Python 3.14 / 3.11  •  Windows 11 (AMD64)  •  Streamlit Community Cloud│
└────────────────────────────────────────────────────────────────────────┘
```

---

### 2. Runtime Environment and Hardware Specifications

#### 2.1 Local Engineering Workstation
- **Operating System:** Microsoft Windows 11 Enterprise (Build 26100, 64-bit AMD64)
- **Host Architecture:** x86_64 / Intel(R) Core(TM) processor architecture
- **System Memory:** 16 GB+ High-Speed DDR4/DDR5 RAM
- **Storage:** Solid-State NVMe Drive (enabling sub-second WESAD multi-megabyte window slicing)
- **Python Runtime:** Python 3.14.0 / 3.11.x (CPython, 64-bit)
- **Shell / CLI Tools:** PowerShell 7, Windows Command Line, Git 2.47+

#### 2.2 Cloud Deployment Target
- **Platform:** Streamlit Community Cloud (Debian Linux Container runtime)
- **Container Path:** `/mount/src/multimodal-threat-prediction/app/main.py`
- **Networking:** Secure HTTPS on port 443 with WebSocket streaming protocol

---

### 3. Core Software Libraries and Installed Versions

The exact installed package versions in the production Python environment are recorded below:

| Package Name | Installed Version | Minimum Version Required | Core Role & Architectural Responsibility |
| :--- | :---: | :---: | :--- |
| **Python** | `3.14.0` | `3.10.0+` | Core programming language runtime |
| **NumPy** | `2.5.2` | `1.24.0` | N-dimensional array processing, signal slicing, vector norms |
| **Pandas** | `3.0.5` | `2.0.0` | Dataframe manipulation, multi-rate temporal joins, CSV/Excel parsing |
| **SciPy** | `1.18.1` | `1.10.0` | Peak detection (`scipy.signal.find_peaks`), skewness, HRV math |
| **Scikit-Learn** | `1.9.0` | `1.2.0` | Random Forest classifiers/regressors, GroupKFold, StandardScaler |
| **XGBoost** | `3.4.1` | `1.7.0` | Gradient boosted decision trees for cost-sensitive crowd prediction |
| **Streamlit** | `1.62.0` | `1.25.0` | Reactive web dashboard, component state management, UI rendering |
| **Matplotlib** | `3.11.1` | `3.7.0` | Publication-quality 300 DPI academic figure rendering |
| **Seaborn** | `0.13.2` | `0.12.0` | Statistical heatmaps, confusion matrices, and distribution KDEs |
| **Joblib** | `1.5.3` | `1.2.0` | Parallelized multi-core training pipelines (`n_jobs=-1`) |
| **OpenPyXL** | `3.1.5` | `3.1.0` | Excel workbook parser for BIT Mesra schedule data |
| **tzdata** | `2025.1` | `2023.3` | Centralized IANA timezone database (`Asia/Kolkata`) |

---

### 4. Official Project `requirements.txt` File Content

The actual `requirements.txt` file committed to the repository root is reproduced verbatim below:

```text
numpy>=1.24.0
pandas>=2.0.0
scipy>=1.10.0
scikit-learn>=1.2.0
xgboost>=1.7.0
matplotlib>=3.7.0
seaborn>=0.12.0
joblib>=1.2.0
streamlit>=1.25.0
streamlit-js-eval>=0.1.5
openpyxl>=3.1.0
tzdata>=2023.3
```

---

### 5. Automated Testing and CI/CD Verification Frameworks
- **Primary Test Runner:** `unittest` (Standard Python Test Framework, fully integrated without external test-runner overhead).
- **Headless UI Testing:** `streamlit.testing.v1.AppTest` for headless simulating of button clicks, selectbox interactions, and session-state mutation.
- **Execution Script:**
  ```bash
  python -m unittest discover tests -v
  ```
- **Test Integrity Constraint:** No proprietary cloud credentials, secret API keys, or external database connections are required to run the full 104-test suite; all tests execute in a sandboxed, fully offline local environment.
