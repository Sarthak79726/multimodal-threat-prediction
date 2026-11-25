# Multimodal Threat Prediction Engine — Version 1.0
# MASTER TECHNICAL REPORT CONTEXT & ACADEMIC EVIDENCE REPOSITORY
## Department of Computer Science & Engineering, Birla Institute of Technology (BIT), Mesra

---

> [!IMPORTANT]
> **PRIMARY CLAUDE AI REPORT GENERATION DOCUMENT:**  
> This master document is a comprehensive, self-contained technical evidence dossier compiled directly from the production repository at `E:\project1`. It encapsulates the entire engineering, mathematical, algorithmic, empirical, and architectural foundation of the **Multimodal Threat Prediction Engine (Version 1.0)**. An external AI assistant can utilize this single verified document to author a rigorous, publication-grade 30-page B.Tech academic project report without requiring access to the 20 GB raw dataset and model binaries.

---

## 1. Project Overview & Institutional Identification
- **Project Title:** Multimodal Threat Prediction Engine
- **Institutional Context:** B.Tech Major Project, Department of Computer Science & Engineering, Birla Institute of Technology (BIT), Mesra, Ranchi, Jharkhand, India.
- **System Version:** Version 1.0 (Audited Baseline Architecture).
- **Core Technology:** Python 3.14/3.11, Scikit-Learn, XGBoost, Streamlit, Folium GIS, SciPy Signal Processing.
- **Primary Operational Goal:** Passively and adaptively evaluate an objective contextual threat and distress index ($R \in [0.0, 1.0]$) by synergistically fusing:
  1. Wearable physiological distress telemetry (WESAD benchmark dataset).
  2. Deterministic geospatial vulnerability across 17 BIT Mesra campus landmarks.
  3. Situational environmental predictions (crowd density, illumination, and lux) derived from campus diurnal schedules.
- **Verification Milestone:** 18 test suites comprising 104 automated tests achieving a **100% pass rate** in $34.80$ seconds.

---

## 2. Problem Statement & Research Objectives

### 2.1 The Problem Statement
Traditional personal safety tools (physical panic buttons, emergency SOS apps) fundamentally rely on the victim's ability to execute a conscious, manual trigger during acute danger. In genuine physical assault, harassment, or severe distress, victims are frequently incapacitated, physically restrained, or physiologically immobilized by shock (the tonic immobility or "freeze" response), rendering manual alerts ineffective.

Conversely, standalone automated monitors fail due to modality-specific limitations:
- **Biometrics Alone:** Cannot distinguish between malevolent terror and benign excitement (vigorous sports, laughter, academic exams), generating high false-alarm rates.
- **Location Alone:** Generates persistent alerts whenever a student walks through secluded campus sectors at night, irrespective of their actual well-being.
- **Environmental Context Alone:** Predicts ambient darkness or low foot traffic, but cannot infer whether an individual is experiencing harm.

### 2.2 Formal Research Objectives
1. **Biometric Distress Classification:** Process raw multi-sensor telemetry from the WESAD dataset across 15 subjects, segment into 60-second sliding windows (10s stride), extract 76 verified statistical/HRV features, and train a subject-independent classifier achieving ROC-AUC $\ge 0.95$.
2. **Deterministic Campus Geospatial Engine:** Model 17 verified BIT Mesra landmarks, calculating great-circle Haversine distances, enforcing a 350m campus geofence, and engineering an automated ambiguity detector that halts automated snapping and enforces manual confirmation when coordinates are equidistant within the GPS accuracy margin.
3. **Environmental Context & Class Balancing:** Synthesize a 180-day diurnal environmental dataset ($N = 73,440$), train multi-class classifiers, and resolve minority transition class suppression in crowd density prediction using cost-sensitive sample weighting ($w_c = \frac{N}{C \cdot N_c}$), improving Medium-class recall from $4.48\%$ to $42.39\%$.
4. **Resilient Multimodal Late Fusion:** Formulate a weighted late-fusion index ($R = 0.40P + 0.25L + 0.35E$) and engineer an un-fabricated partial dual-branch fallback ($R_{\text{dual}} = \frac{0.40P + 0.25L}{0.65}$) when environmental inputs are unavailable, eliminating silent neutral imputation.
5. **Interactive UI & Ethical Governance:** Deploy a responsive Streamlit dashboard featuring Folium mapping, unrounded telemetry formatting, prominent provenance watermarks, and a strict human-in-the-loop ethical governance framework.

---

## 3. Complete System Architecture & High-Level Workflow

### 3.1 Structural System Flow
The engine operates across five sequential stages:
1. **Sensing Layer:** Ingests chest/wrist wearable signals (WESAD), browser GPS coordinates (HTML5 Geolocation API), and diurnal calendar time (IST).
2. **Feature Extraction Layer:** Converts raw signals into 76 physiological features, calculates Haversine distances to 17 campus landmarks, and encodes diurnal environmental features (hour, weekday, weekend).
3. **Domain Inference Layer:**
   - Physiological Branch: Random Forest (150 trees) $\to P_{\text{physio}} \in [0.0, 1.0]$.
   - Spatial Branch: Deterministic Geospatial Engine $\to P_{\text{spatial}} \in [0.0, 1.0]$.
   - Environmental Branch: Balanced XGBoost (Crowd) + Random Forest (Lighting & Lux) $\to P_{\text{env}} \in [0.05, 1.00]$.
4. **Multimodal Fusion & Missing-Data Layer:** Dynamically routes data through complete 3-branch weighted fusion ($0.40P + 0.25L + 0.35E$) or renormalized partial dual-branch fusion ($(0.40P + 0.25L)/0.65$) based on sensor availability.
5. **Presentation & Decision Support Layer:** Streamlit dashboard renders the overall threat index (`LOW`: $\le 0.39$, `MEDIUM`: $0.40-0.69$, `HIGH`: $\ge 0.70$), Folium map with radar range, dynamic contribution breakdown, and provenance disclosures.

---

## 4. Empirical Datasets & Feature Engineering

### 4.1 WESAD Benchmark Dataset Specifications
- **Source:** Ubiquitous Computing Lab, University of Siegen, Germany (Schmidt et al., ICMI 2018).
- **Population:** 15 usable human subjects (`S2` through `S17`, excluding `S1` and `S12` per protocol).
- **Hardware & Channels:**
  - RespiBAN Chest ($700\,\text{Hz}$): ECG ($\text{mV}$), EDA ($\mu\text{S}$), EMG ($\text{mV}$), Skin Temp ($^\circ\text{C}$), Respiration ($\text{V}$), 3-Axis ACC ($\text{g}$).
  - Empatica E4 Wrist: BVP ($64\,\text{Hz}$), EDA ($4\,\text{Hz}$), Skin Temp ($4\,\text{Hz}$), 3-Axis ACC ($32\,\text{Hz}$).
- **Target Formulation:** Binary classification between Stress (Label 2, TSST protocol) and Non-Stress (Label 1 Baseline, Label 3 Amusement, Label 4 Meditation). Labels 0, 5, 6, 7 excluded.

### 4.2 The 76-Dimensional Physiological Feature Vector
- **Window Parameters:** $60.0\,\text{s}$ window ($42,000$ chest samples), $10.0\,\text{s}$ stride ($7,000$ samples), $83.33\%$ overlap.
- **Statistical Features (per channel):** Mean, standard deviation, minimum, maximum, skewness, linear trend slope.
- **Accelerometry Features (Chest & Wrist):** 3-axis Euclidean magnitude $\|\mathbf{a}\| = \sqrt{a_x^2 + a_y^2 + a_z^2}$, magnitude standard deviation, magnitude peak, and per-axis means/stds.
- **Heart Rate Variability (HRV) Features (Chest ECG & Wrist BVP):**
  - QRS / Systolic Peak Detection: Minimum peak distance $d_{\min} = f_s \times 0.3\,\text{s}$ (max $200\,\text{BPM}$).
  - Heart Rate: $\text{BPM} = \frac{60.0}{\mu_{\text{RR}}}$.
  - Time-Domain Metrics: Mean RR interval ($\mu_{\text{RR}}$), SDNN ($\sigma_{\text{RR}}$), and RMSSD:
    $$\text{RMSSD} = \sqrt{\frac{1}{N-1} \sum_{i=1}^{N-1} (RR_{i+1} - RR_i)^2}$$
- **Total Dataset Size:** 4,325 extracted window vectors (3,365 Non-Stress, 960 Stress).

### 4.3 BIT Mesra Geographic Survey (17 Landmarks)
Central Benchmark: Main Building Clock Tower ($23.412300^\circ\,\text{N}, 85.439900^\circ\,\text{E}$).
1. `Main Building`: $23.412300, 85.439900$, $r=150\text{m}$, Administrative Hub, $R_0=0.08, I_0=0.05$, Sec Dist: $20\text{m}$
2. `R&D / Library`: $23.413500, 85.438500$, $r=150\text{m}$, Academic Zone, $R_0=0.12, I_0=0.15$, Sec Dist: $50\text{m}$
3. `Hostel 1 (H1)`: $23.414209, 85.440313$, $r=120\text{m}$, Residential, $R_0=0.25, I_0=0.25$, Sec Dist: $120\text{m}$
4. `Hostel 2 (H2)`: $23.413707, 85.438484$, $r=120\text{m}$, Residential, $R_0=0.25, I_0=0.25$, Sec Dist: $130\text{m}$
5. `Hostel 3 (H3)`: $23.416120, 85.439653$, $r=120\text{m}$, Residential, $R_0=0.28, I_0=0.30$, Sec Dist: $150\text{m}$
6. `Hostel 4 (H4)`: $23.415602, 85.437798$, $r=120\text{m}$, Residential, $R_0=0.28, I_0=0.30$, Sec Dist: $170\text{m}$
7. `Girls Hostel / H8-H9`: $23.414000, 85.441500$, $r=150\text{m}$, Residential, $R_0=0.20, I_0=0.20$, Sec Dist: $80\text{m}$
8. `PMC`: $23.416205, 85.440121$, $r=150\text{m}$, Unlit Industrial Outpost, $R_0=0.65, I_0=0.80$, Sec Dist: $480\text{m}$
9. `Hostel 6 (H6)`: $23.422811, 85.433202$, $r=120\text{m}$, Residential, $R_0=0.30, I_0=0.35$, Sec Dist: $200\text{m}$
10. `Hostel 7 (H7)`: $23.424604, 85.432130$, $r=120\text{m}$, Residential, $R_0=0.32, I_0=0.40$, Sec Dist: $230\text{m}$
11. `Dispensary`: $23.410500, 85.438800$, $r=100\text{m}$, Medical Center, $R_0=0.15, I_0=0.18$, Sec Dist: $100\text{m}$
12. `Hostel 10 (H10)`: $23.419115, 85.434912$, $r=120\text{m}$, Isolated Boundary, $R_0=0.45, I_0=0.70$, Sec Dist: $400\text{m}$
13. `Hostel 11 (H11)`: $23.418032, 85.435418$, $r=120\text{m}$, Isolated Boundary, $R_0=0.50, I_0=0.78$, Sec Dist: $450\text{m}$
14. `Hostel 12 (H12)`: $23.418919, 85.434311$, $r=120\text{m}$, Isolated Boundary, $R_0=0.55, I_0=0.85$, Sec Dist: $520\text{m}$
15. `Hostel 13 (H13)`: $23.417810, 85.434655$, $r=120\text{m}$, Isolated Boundary, $R_0=0.60, I_0=0.92$, Sec Dist: $600\text{m}$
16. `Sports Complex`: $23.425020, 85.434623$, $r=200\text{m}$, Open Recreational, $R_0=0.35, I_0=0.55$, Sec Dist: $250\text{m}$
17. `Main Gate`: $23.405500, 85.435000$, $r=150\text{m}$, Guarded Checkpoint 1, $R_0=0.05, I_0=0.02$, Sec Dist: $0\text{m}$

### 4.4 Synthetic Environmental Simulation Dataset
- **Generation:** 180 simulated days $\times$ 17 locations $\times$ 24 hourly slots = 73,440 records (`seed=42`).
- **Splits:** Train (Days 1–120: 48,960), Val (Days 121–150: 12,240), Test (Days 151–180: 12,240).
- **Distributions:**
  - Crowd: High (48.19%), Low (30.13%), Medium (21.68%).
  - Lighting: Good (71.00%), Low (16.99%), Dark (12.01%).
  - Illuminance: Mean = $1,288.75\,\text{lx}$, Range = $0.10$ to $9,450.00\,\text{lx}$.

---

## 5. Mathematical Formulations & Decision Logic

### 5.1 Haversine Great-Circle Distance
$$d = 2R \arcsin \left( \sqrt{\sin^2\left(\frac{\Delta \phi}{2}\right) + \cos(\phi_1)\cos(\phi_2)\sin^2\left(\frac{\Delta \lambda}{2}\right)} \right), \quad R = 6,371,000.0\,\text{m}$$

### 5.2 Campus Geofencing & Ambiguity Detection
- **Geofence Check:** $d_1 = \min d(\mathbf{x}, \text{Landmarks})$. If $d_1 > 350.0\,\text{m} \implies \text{OUTSIDE\_CAMPUS\_GEOFENCE}$.
- **Ambiguity Test:** Sort landmarks $d_1 \le d_2 \le \dots \le d_{17}$.
  $$\text{If } |d_2 - d_1| \le \max(25.0\,\text{m}, \text{accuracy}) \implies \text{AMBIGUOUS\_LOCATION}$$

### 5.3 Deterministic Spatial Vulnerability
$$d_{\text{center}} = \text{haversine}(\phi, \lambda, 23.4123, 85.4399)$$
$$I_{\text{spatial}} = \min\left(1.0, \max\left(0.0, I_0 + \frac{d_{\text{center}}}{1000.0} \times 0.2\right)\right)$$
$$P_{\text{night}} = 0.25 \text{ if (hour} \ge 19.0 \lor \text{hour} \le 5.0) \text{ else } 0.00$$
$$P_{\text{spatial}} = \min(1.0, \max(0.0, R_0 + P_{\text{night}} \times I_{\text{spatial}}))$$

### 5.4 Situational Environmental Score
$$P_{\text{env}} = \min(1.0, \max(0.05, \text{Factor}_{\text{time}} + \text{Factor}_{\text{light}} + \text{Factor}_{\text{crowd}}))$$
- $\text{Factor}_{\text{time}}$: Night ($0.25$), Twilight ($0.12$), Day ($0.05$).
- $\text{Factor}_{\text{light}}$: Dark ($0.35$), Low ($0.18$), Good ($0.05$).
- $\text{Factor}_{\text{crowd}}$: Low ($0.32$), Medium ($0.15$), High ($0.02$).

### 5.5 Multimodal Late-Fusion Formulations
- **Complete Three-Branch Formulation:**
  $$R = 0.40 \cdot P_{\text{physio}} + 0.25 \cdot P_{\text{spatial}} + 0.35 \cdot P_{\text{env}}$$
- **Partial Dual-Branch Renormalized Formulation (Missing E):**
  $$R_{\text{dual}} = \frac{0.40 \cdot P_{\text{physio}} + 0.25 \cdot P_{\text{spatial}}}{0.65} \approx 0.6154 \cdot P_{\text{physio}} + 0.3846 \cdot P_{\text{spatial}}$$

---

## 6. Comprehensive Model Benchmarks & Empirical Findings

### 6.1 Physiological Models Benchmark (WESAD 5-Fold GroupKFold OOF, N = 4,325)
1. **Logistic Regression (Standardized):** Accuracy = $85.11\%$, Precision = $67.59\%$, Recall = $63.23\%$, F1 = $0.6534$, ROC-AUC = $0.8057$.
2. **Gradient Boosting (XGBoost):** Accuracy = $91.42\%$, Precision = $84.20\%$, Recall = $75.52\%$, F1 = $0.7963$, ROC-AUC = $0.9628$.
3. **Random Forest Classifier (Selected):** **Accuracy = $93.66\%$**, **Precision = $92.03\%$**, **Recall = $78.23\%$**, **F1 = $0.8457$**, **ROC-AUC = $0.9766$**.
   - Confusion Matrix: $TN = 3,300, FP = 65, FN = 209, TP = 751$. Specificity = $98.07\%$.

### 6.2 Crowd Density Models Benchmark (Chronological Test Set, N = 12,240)
1. **Dummy Classifier:** Accuracy = $48.06\%$, Macro F1 = $0.2164$, Medium Recall = $0.00\%$.
2. **Logistic Regression (Balanced):** Accuracy = $44.22\%$, Macro F1 = $0.4125$, Medium Recall = $43.68\%$.
3. **Random Forest (Balanced):** Accuracy = $71.91\%$, Macro F1 = $0.6499$, Medium Recall = $38.14\%$.
4. **Unweighted XGBoost Baseline:** Accuracy = $72.67\%$, Macro F1 = $0.5702$, Medium Recall = $4.48\%$.
5. **Cost-Sensitive Balanced XGBoost (Selected):** **Accuracy = $71.32\%$**, **Macro Precision = $0.6693$**, **Macro Recall = $0.6761$**, **Macro F1 = $0.6722$**, **Medium Recall = $42.39\%$** (~$9.5\times$ gain).
   - Test Confusion Matrix:
     - True Low: $[3112, 293, 296]$ (Recall = $84.09\%$)
     - True Med: $[715, 1126, 815]$ (Recall = $42.39\%$)
     - True High: $[141, 1250, 4492]$ (Recall = $76.36\%$)

### 6.3 Lighting Models Benchmark (Chronological Test Set, N = 12,240)
1. **Dummy Classifier:** Accuracy = $70.95\%$, Macro F1 = $0.2767$.
2. **Logistic Regression:** Accuracy = $73.42\%$, Macro F1 = $0.3957$.
3. **Random Forest Classifier (Selected):** **Accuracy = $88.46\%$**, **Macro F1 = $0.7833$** (Dark Recall = $62.37\%$, Low Recall = $64.91\%$, Good Recall = $98.54\%$).
   - Test Confusion Matrix:
     - True Dark: $[915, 460, 92]$
     - True Low: $[142, 1356, 591]$
     - True Good: $[23, 104, 8557]$
4. **Continuous Lux Regressor (Random Forest):** Test MAE = **$378.10\,\text{lx}$**, Test RMSE = **$650.54\,\text{lx}$**, **$R^2 = 0.7571$**.

---

## 7. Software Verification & Quality Assurance Audit
- **Test Suite Results:** 18 distinct test files, 104 executed tests, **104 passed (100.0%)**, 0 failed, 0 errors, 0 skipped.
- **Execution Date:** November 25, 2026 ($34.800\,\text{seconds}$).
- **Resolved Bugs Verified by Tests:**
  1. Resolved `resolve_campus_location` TypeError on Streamlit Cloud via defensive kwargs and try/except fallbacks.
  2. Fixed meditation 50% probability artifact by engineering `find_best_contiguous_window()`.
  3. Eliminated baseline and amusement integer truncation, restoring unrounded decimal probabilities.
  4. Eliminated silent neutral substitution in late fusion via explicit weight renormalization.

---

## 8. Summary of Figures and Evidence Artifacts Available in Package
The accompanying package contains:
- **26 Publication-Quality Academic Figures (`FIGURES/`):** Covering architecture, data flow, pipelines, decision trees, confusion matrices, class balancing comparisons, lux regressions, heatmaps, and future IoT roadmaps.
- **Lightweight Source Code (`SOURCE_CODE/`):** 58 source files (519.9 KB) spanning `app/`, `src/`, `tests/`, `config.py`, `requirements.txt`.
- **Dataset Summaries (`DATASET_SUMMARIES/`):** Landmark catalogs, schema files, distributions, hourly patterns, and 76-feature dictionary.
- **Experimental Artifacts (`EXPERIMENTAL_EVIDENCE/`):** Metrics JSONs, CSV comparisons, and WESAD subject audits.
- **Screenshot Capture Guide (`DASHBOARD_SCREENSHOTS/`):** Instructions and specifications for capturing live dashboard views.
