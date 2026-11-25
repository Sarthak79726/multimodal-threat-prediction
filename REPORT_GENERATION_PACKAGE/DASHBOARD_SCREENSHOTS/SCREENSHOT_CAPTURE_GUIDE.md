# Dashboard Screenshot Capture Guide

This guide describes how to capture live screenshots from your local Streamlit dashboard or Streamlit Cloud deployment for inclusion in the B.Tech academic report.

## How to Launch the Local Streamlit Dashboard

`ash
# Activate your virtual environment
.venv\Scripts\activate

# Launch Streamlit
streamlit run app/main.py
`
The application will open in your browser at http://localhost:8501.

---

## 8 Recommended Academic Report Screenshots

### 1. Main Dashboard (ui01_main_dashboard.png)
- **View:** Tab 1: Multimodal Threat Prediction Dashboard.
- **State:** Initial state after loading, displaying the Header, 3 Branch Cards (Physiological, Spatial, Environmental), and the Primary Evaluation Button.
- **Caption:** Figure 9.1: Multimodal Threat Prediction Engine — Main Streamlit Dashboard Interface.

### 2. Physiological Analysis Branch (ui02_physiological_analysis.png)
- **View:** Column 1 (Physiological Analysis Card).
- **Settings:** Select Subject S2, Condition Stress (Label 2).
- **Output:** Displays Stress Probability (e.g. 99.8% / HIGH RISK), ECG Mean, Heart Rate (BPM), RMSSD, and 60s signal waveform preview.
- **Caption:** Figure 9.2: Physiological Stress Detection Interface showing WESAD S2 Stress Condition telemetry.

### 3. Spatial Context & GPS Analyzer (ui03_spatial_analysis.png)
- **View:** Tab 2: BIT Mesra GPS & Spatial Analyzer.
- **Settings:** Location query for Main Building or H13.
- **Output:** Nearest Campus Landmark, Distance to Landmark (m), Geofence Status, Distance to Security Post, and Interactive Folium Leaflet Map with radar circle.
- **Caption:** Figure 9.3: BIT Mesra Campus GPS & Geospatial Map Analyzer.

### 4. Environmental AI Predictions (ui04_environmental_predictions.png)
- **View:** Column 3 (Environmental Context Card).
- **Settings:** Landmark Main Building, Time 14:00 IST (2:00 PM).
- **Output:** Predicted Crowd: High (XGBoost Balanced), Predicted Lighting: Good (~2200 lx, Random Forest), Situational Environmental Score.
- **Caption:** Figure 9.4: Environmental AI Context Prediction Card showing crowd and lighting estimates.

### 5. Complete Three-Branch Fusion Result (ui05_complete_fusion_output.png)
- **View:** Overall Contextual Risk Level Panel.
- **Settings:** Physiological (0.85), Spatial (0.60), Environmental (0.42).
- **Output:** Threat Score: 68% (MEDIUM RISK), Contribution breakdown (Physiological 50%, Spatial 22%, Environmental 28%), Alert recommendation.
- **Caption:** Figure 9.5: Complete Three-Branch Late Fusion Threat Assessment Panel.

### 6. GPS Ambiguity Handling (ui06_gps_ambiguity_handling.png)
- **View:** GPS Resolution section when coordinates are equidistant (e.g. between H10 and H12).
- **Output:** Warning Banner: AMBIGUOUS GPS LOCATION DETECTED. Position is within accuracy margin between: H10, H12. Please manually confirm your campus location.
- **Caption:** Figure 9.6: System detection of ambiguous GPS coordinates requiring user confirmation.

### 7. Missing-Data & Partial Dual-Branch Fusion (ui07_missing_data_handling.png)
- **View:** Overall Panel when Environmental branch is UNKNOWN / Unavailable.
- **Output:** Warning Banner: Complete 3-branch index is unavailable. The engine evaluates a Partial Dual-Branch Context Index (Physiological + Spatial only). No neutral values are silently substituted. Threat Score calculated using renormalized formula.
- **Caption:** Figure 9.7: Transparent handling of unavailable environmental sensor data via renormalized dual-branch fusion.

### 8. Manual Landmark Override (ui08_manual_landmark_selection.png)
- **View:** Column 2 or 3 checkbox Override GPS Location.
- **Output:** Dropdown menu selecting H13 (Perimeter) overriding live GPS coordinates.
- **Caption:** Figure 9.8: User manual landmark override interface ensuring user agency and fallback resilience.
