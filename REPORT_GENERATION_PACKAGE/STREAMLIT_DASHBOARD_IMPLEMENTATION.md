# Multimodal Threat Prediction Engine — Version 1.0
## Technical Documentation: Streamlit Cloud Dashboard Implementation and User Interface Design

---

### 1. Application Overview and Technology Stack
The user interface layer is built with **Streamlit** (Python web framework), styled with high-contrast medical/academic CSS (`app/assets/css/style.css`), and augmented with custom HTML5/JavaScript components for browser geolocation (`app/components/gps/`) and interactive geospatial mapping via **Folium / Leaflet.js** (`app/components/map/map_component.py`).

- **Entry Point:** `app/main.py` (1,394 lines of defensive Python logic)
- **Execution Port:** `8501` (Default Streamlit port)
- **Target Environments:** Local developer workstation (Windows 11) and Streamlit Community Cloud (`/mount/src/multimodal-threat-prediction/app/main.py`).

---

### 2. User Interface Layout and Structural Hierarchy

The dashboard is structured into a persistent header and two primary operational tabs:

```text
┌────────────────────────────────────────────────────────────────────────┐
│  🛡️ MULTIMODAL THREAT PREDICTION ENGINE — BIT MESRA (Version 1.0)       │
│  [Status: OPERATIONAL]  [IST Time: 02:15 PM]  [Mode: Three-Branch]     │
├────────────────────────────────────────────────────────────────────────┤
│  [Tab 1: 🛡️ Threat Prediction Dashboard]  [Tab 2: 📍 Spatial Analyzer]  │
├────────────────────────────────────────────────────────────────────────┤
│  TAB 1 CONTENTS:                                                       │
│  ┌───────────────────┬───────────────────┬───────────────────────────┐ │
│  │ CARD 1:           │ CARD 2:           │ CARD 3:                   │ │
│  │ Physiological     │ Spatial Context   │ Environmental Context     │ │
│  │ • Subject [S2-S17]│ • GPS Resolver    │ • Hour of Day [0-23]      │ │
│  │ • Condition Pick  │ • Nearest LM      │ • Crowd (XGBoost Balanced)│ │
│  │ • Stress P (%-age)│ • Distance (m)    │ • Lighting (Random Forest)│ │
│  │ • Waveform Chart  │ • Manual Override │ • Continuous Lux (RF)     │ │
│  │ • Biometrics Tbl  │ • Security Dist   │ • Situational Env Score   │ │
│  └───────────────────┴───────────────────┴───────────────────────────┘ │
│                                                                        │
│  [ 🛡️ ANALYZE MULTIMODAL THREAT (Primary Action Button) ]               │
│                                                                        │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │ OVERALL THREAT CONTEXT LEVEL PANEL                                │ │
│  │ • Threat Composite Score: [ 68% MEDIUM RISK ]                     │ │
│  │ • Dynamic Modality Contribution: P: 50% | L: 22% | E: 28%         │ │
│  │ • Modality Status: P: Active | L: Resolved | E: Active (Synthetic)│ │
│  │ • Warning Banners (Missing E / Ambiguous GPS, if applicable)      │ │
│  └───────────────────────────────────────────────────────────────────┘ │
│                                                                        │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │ 🔒 CAMPUS SAFETY, PRIVACY & ETHICAL GOVERNANCE NOTICE             │ │
│  │ • Informed Consent & Wearable Biometric Protection                │ │
│  │ • Human-in-the-Loop Constraint (No Automated Accusations)         │ │
│  └───────────────────────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────────────────────┘
```

---

### 3. Detailed Component Breakdown (Tab 1: Multimodal Dashboard)

#### 3.1 Card 1: Physiological Stress Detection (WESAD Wearables)
- **Subject Selector:** `st.selectbox("Select WESAD Subject", ALL_SUBJECTS)` enabling individual selection across all 15 usable subjects (`S2` through `S17`).
- **Condition Selector:** Allows choosing between `Baseline (Resting)`, `Stress (TSST)`, `Amusement (Humor)`, and `Meditation (Relaxation)`.
- **Data Isolation:** Loads condition telemetry from `data/physiological/wesad_condition_audit.json` with unique SHA-256 fingerprints, guaranteeing distinct biometric vectors.
- **Biometric Telemetry Display:** Formats unrounded probabilities (`f"{score * 100:.2f}%"`) and displays Mean ECG ($\text{mV}$), Heart Rate ($\text{BPM}$), HRV RMSSD ($\text{s}$), Mean EDA ($\mu\text{S}$), and Trapezius EMG ($\text{mV}$).
- **Waveform Preview:** Renders an inline SVG/HTML preview of the 60-second raw ECG and EDA signal windows.

#### 3.2 Card 2: Spatial Context Analysis (BIT Mesra GIS)
- **Live GPS Telemetry:** Displays latitude, longitude, and accuracy margin from browser geolocation.
- **Geofence & Ambiguity Banners:**
  - If ambiguous: Displays amber alert indicating coordinates are equidistant between candidate landmarks.
  - If outside campus: Displays warning that location exceeds the $350\,\text{m}$ geofence.
- **Manual Location Override:** Checkbox `⚙️ Override GPS Location`. When checked, displays a selectbox of all 17 supported campus landmarks, overriding GPS coordinates for downstream threat fusion.

#### 3.3 Card 3: Environmental AI Context (Crowd & Lighting Prediction)
- **Diurnal Time Input:** Slider and numeric input allowing continuous hour selection ($0.0$ to $23.0$ IST).
- **Automated AI Predictions:**
  - **Crowd Density:** Displays category badge (`High`, `Medium`, `Low`), confidence percentage, and model pill (`Gradient Boosting (XGBoost Balanced)`).
  - **Lighting Condition:** Displays category badge (`Good`, `Low`, `Dark`), model pill (`Random Forest`), and continuous lux estimate (`~2,380 lx`).
- **Situational Environmental Score:** Calculated deterministically from time, lighting, and crowd factors.
- **Data Provenance Banner:** Displays prominent watermark:
  > *"⚡ EXPERIMENTAL PREDICTION — NOT LIVE SENSOR MEASUREMENT. Actual campus crowd and lighting conditions may differ from predictions."*

#### 3.4 Overall Threat Assessment Panel & Contribution Breakdown
- **Composite Context Index:** Rendered with color-coded severity:
  - Green (`score-safe`): $\le 39\%$ (LOW RISK)
  - Amber (`score-warn`): $40\% - 69\%$ (MEDIUM RISK)
  - Red (`score-danger`): $\ge 70\%$ (HIGH RISK)
- **Missing Modality Banner:** When environmental input is unavailable, the panel displays an amber warning confirming that the system is operating in **Partial Dual-Branch Mode** without silent imputation.
- **Modality Contribution Breakdown:** Dynamic bar visualization illustrating the relative percentage influence of each branch.

---

### 4. Tab 2: BIT Mesra GPS & Geospatial Map Analyzer
- **Interactive Folium Leaflet Map:**
  - Centered at campus centroid ($23.4123^\circ\,\text{N}, 85.4399^\circ\,\text{E}$).
  - Pins all 17 surveyed landmarks with distinct color-coded markers (Hostels: Blue, Academic: Green, Perimeter: Red, Security: Gold).
  - Renders a semi-transparent circular radar radius representing the user's reported location and GPS accuracy margin.
- **Geospatial Metrics Grid:**
  - Distance to nearest landmark (meters).
  - Distance to 24/7 guarded security outpost (meters).
  - Derived spatial isolation index ($[0.0, 1.0]$).
  - Time-adjusted illumination score ($[0.0, 10.0]$).

---

### 5. Streamlit Session State Management

| Session State Key | Data Type | Default Value | Purpose |
| :--- | :--- | :---: | :--- |
| `selected_subject_id` | `string` | `'S2'` | Currently active WESAD subject |
| `selected_condition_name` | `string` | `'Baseline (Label 1)'` | Currently active affective condition |
| `selected_landmark_name` | `string` | `'Main Building'` | Manually selected or overridden landmark |
| `analyzed_threat` | `dict` or `None` | `None` | Output payload from `predict_fusion()` |
| `location_res` | `dict` or `None` | `None` | GPS coordinates and accuracy from browser |
| `env_score` | `float` or `None` | `0.10` | Situational environmental score |
| `client_hour` | `float` or `None` | `None` | Centralized IST time hour float |
| `location_mode` | `string` | `'INITIALIZING'`| State machine status for location resolver |

---

### 6. Local Dashboard Launch and Operational Verification

To execute and verify the dashboard on a local workstation:

```bash
# 1. Open terminal at project root
cd E:\project1

# 2. Activate virtual environment
.venv\Scripts\activate

# 3. Launch Streamlit application
streamlit run app/main.py --server.port 8501
```

Once launched, navigate to `http://localhost:8501` in your browser. All three branch cards, Folium map layers, and reactive threat evaluation callbacks operate instantaneously in bare mode.
