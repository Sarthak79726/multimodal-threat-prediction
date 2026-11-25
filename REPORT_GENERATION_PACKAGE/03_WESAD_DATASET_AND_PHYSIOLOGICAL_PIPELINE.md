# Multimodal Threat Prediction Engine — Version 1.0
## Technical Documentation: WESAD Benchmark Dataset and Physiological Stress Analysis Pipeline

---

### 1. WESAD Dataset Description and Empirical Protocol
The physiological distress branch is built upon the **Wearable Stress and Affect Detection (WESAD)** benchmark dataset, developed by Schmidt et al. (Ubomp 2018) at the Ubiquitous Computing Lab, University of Siegen, Germany.

WESAD is a publicly available, rigorously validated multimodal affective dataset designed to facilitate machine learning research in wearable biometric monitoring. During data collection, subjects underwent a structured multi-condition laboratory protocol lasting approximately two hours per subject:
1. **Baseline Phase (Label 1):** Subjects sat comfortably reading neutral magazines for 20 minutes to establish individual resting physiological baselines.
2. **Stress Phase (Label 2):** Acute somatic and cognitive stress was induced using the standardized **Trier Social Stress Test (TSST)**, comprising a 5-minute public speaking preparation, a 5-minute presentation in front of a committee, and a 5-minute mental arithmetic task (counting backwards from 2043 in steps of 17 with acoustic error feedback).
3. **Amusement Phase (Label 3):** Subjects viewed a sequence of eleven humorous video clips (total duration ~20 minutes) to elicit positive emotional arousal distinct from distress.
4. **Meditation Phase (Label 4):** Guided breathing and relaxation exercises were conducted across two sessions to observe parasympathetic recovery.
5. **Transient / Transition Periods (Labels 0, 5, 6, 7):** Uncontrolled intervals between experimental conditions.

#### Subject Population and Exclusion Protocol
- **Total Dataset Size:** 17 recorded human subjects.
- **Usable Subjects ($N = 15$):** `S2`, `S3`, `S4`, `S5`, `S6`, `S7`, `S8`, `S9`, `S10`, `S11`, `S13`, `S14`, `S15`, `S16`, and `S17`.
- **Excluded Subjects:** `S1` (preliminary pilot protocol without standardized timing) and `S12` (severe sensor hardware synchronization and packet loss during collection), adhering strictly to WESAD author specifications.

---

### 2. Sensor Modalities, Placement, and Sampling Rates

The dataset captures dual-location synchronized biometric signals:

| Sensor Device | Anatomical Placement | Modality | Sampling Frequency | Physical Unit | Description / Physiological Correlate |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **RespiBAN Professional** | Chest (Thoracic Band) | **ECG** | $700\,\text{Hz}$ | $\text{mV}$ | Lead II Electrocardiogram (Heart Rate, HRV, Sympathovagal Balance) |
| **RespiBAN Professional** | Chest (Thoracic Band) | **EDA** | $700\,\text{Hz}$ | $\mu\text{S}$ | Electrodermal Activity / Galvanic Skin Response (Sympathetic Arousal) |
| **RespiBAN Professional** | Chest (Thoracic Band) | **EMG** | $700\,\text{Hz}$ | $\text{mV}$ | Electromyogram of Trapezius Muscle (Somatic Muscular Tension) |
| **RespiBAN Professional** | Chest (Thoracic Band) | **TEMP** | $700\,\text{Hz}$ | $^\circ\text{C}$ | Skin Temperature (Peripheral Vasoconstriction during Stress) |
| **RespiBAN Professional** | Chest (Thoracic Band) | **RESP** | $700\,\text{Hz}$ | $\text{V}$ | Respiration Chest Expansion (Breathing Rate, Hyperventilation) |
| **RespiBAN Professional** | Chest (Thoracic Band) | **ACC** | $700\,\text{Hz}$ | $\text{g}$ | 3-Axis Accelerometer (X, Y, Z Chest Motion and Posture) |
| **Empatica E4** | Non-Dominant Wrist | **BVP** | $64\,\text{Hz}$ | $\text{unitless}$ | Blood Volume Pulse via Photoplethysmography (PPG Pulse Waves) |
| **Empatica E4** | Non-Dominant Wrist | **EDA** | $4\,\text{Hz}$ | $\mu\text{S}$ | Wrist Electrodermal Conductance (Palmar/Wrist Sweat Glands) |
| **Empatica E4** | Non-Dominant Wrist | **TEMP** | $4\,\text{Hz}$ | $^\circ\text{C}$ | Peripheral Wrist Skin Temperature |
| **Empatica E4** | Non-Dominant Wrist | **ACC** | $32\,\text{Hz}$ | $\text{g}$ | 3-Axis Wrist Accelerometer (Tremors, Hand Agitation, Locomotion) |

---

### 3. Signal Preprocessing, Windowing, and Label Formulation

#### 3.1 Windowing and Segmentation Parameters
- **Window Size ($T_w$):** $60.0\,\text{seconds}$ ($42,000$ samples at chest rate $700\,\text{Hz}$). A 60-second window is the established standard in physiological signal processing, providing sufficient duration for statistically meaningful Heart Rate Variability (HRV) time-domain metrics.
- **Window Stride ($T_s$):** $10.0\,\text{seconds}$ ($7,000$ samples at chest rate $700\,\text{Hz}$).
- **Window Overlap:** $83.33\%$ ($\frac{50\,\text{s}}{60\,\text{s}}$), providing temporal continuity while preventing boundary truncation artifacts.

#### 3.2 Ground-Truth Target Label Formulation
The project solves a **Binary Distress Classification** problem:
- **Class 1 (Stress):** Label 2 (TSST Stress Phase).
- **Class 0 (Non-Stress):** Label 1 (Resting Baseline), Label 3 (Amusement), and Label 4 (Meditation).
- **Excluded Labels:** Label 0 (transient period between tasks) and Labels 5, 6, 7 (non-experimental intervals).

**Dominant Label Rule:** For any window $[t, t + T_w]$, the raw 700 Hz label array is mapped to binary classes. A window is retained if at least $70\%$ of its samples belong to valid experimental labels, and the majority binary class determines the ground-truth target.

#### 3.3 Contiguous Window Selection for Meditation
To eliminate the 50% probability artifact identified during project audits, `find_best_contiguous_window()` searches the subject's two split meditation sessions to locate the longest contiguous session, selecting the central 60 seconds where transient noise is strictly $0\%$.

---

### 4. Comprehensive 76-Dimensional Physiological Feature Set

A total of **76 verified physiological features** are extracted for every 60-second window across all chest and wrist channels.

#### 4.1 Complete Feature Directory by Modality

| Index Range | Feature Names | Sensor & Placement | Mathematical Definition & Extraction Logic | Physical Unit |
| :--- | :--- | :--- | :--- | :--- |
| **1–6** | `chest_ecg_mean`, `chest_ecg_std`, `chest_ecg_min`, `chest_ecg_max`, `chest_ecg_skew`, `chest_ecg_slope` | Chest ECG ($700\,\text{Hz}$) | 1st–3rd statistical moments, range extrema, and 1st-order linear regression fit slope | $\text{mV}$ |
| **7–11** | `chest_ecg_peak_count`, `chest_ecg_hr_bpm`, `chest_ecg_rr_mean`, `chest_ecg_rr_std`, `chest_ecg_rmssd` | Chest ECG ($700\,\text{Hz}$) | QRS peak detection ($d_{\min} = 0.3\,\text{s}$), instantaneous BPM ($\frac{60}{\mu_{\text{RR}}}$), SDNN ($\sigma_{\text{RR}}$), and RMSSD ($\sqrt{\frac{1}{N}\sum (\Delta \text{RR})^2}$) | $\text{count}, \text{BPM}, \text{s}, \text{s}, \text{s}$ |
| **12–17** | `chest_eda_mean`, `chest_eda_std`, `chest_eda_min`, `chest_eda_max`, `chest_eda_skew`, `chest_eda_slope` | Chest EDA ($700\,\text{Hz}$) | Skin conductance baseline, phasic variance, dynamic range, distributional asymmetry, and trend | $\mu\text{S}$ |
| **18–23** | `chest_emg_mean`, `chest_emg_std`, `chest_emg_min`, `chest_emg_max`, `chest_emg_skew`, `chest_emg_slope` | Chest EMG ($700\,\text{Hz}$) | Trapezius muscle action potentials, tonic activation variance, peak contraction spikes, and fatigue slope | $\text{mV}$ |
| **24–29** | `chest_temp_mean`, `chest_temp_std`, `chest_temp_min`, `chest_temp_max`, `chest_temp_skew`, `chest_temp_slope` | Chest Temp ($700\,\text{Hz}$) | Thoracic epidermal temperature level, stability, thermal drift, and linear cooling/warming rate | $^\circ\text{C}$ |
| **30–35** | `chest_resp_mean`, `chest_resp_std`, `chest_resp_min`, `chest_resp_max`, `chest_resp_skew`, `chest_resp_slope` | Chest Resp ($700\,\text{Hz}$) | Thoracic expansion volume baseline, tidal volume variance, respiratory range, and inhalation trend | $\text{V}$ |
| **36–44** | `chest_acc_mag_mean`, `chest_acc_mag_std`, `chest_acc_mag_max`, `chest_acc_x_mean`, `chest_acc_x_std`, `chest_acc_y_mean`, `chest_acc_y_std`, `chest_acc_z_mean`, `chest_acc_z_std` | Chest ACC ($700\,\text{Hz}$) | 3-axis Euclidean norm $\|\mathbf{a}\| = \sqrt{a_x^2 + a_y^2 + a_z^2}$, motion variance, peak jerk, and individual orthometric axis projections | $\text{g}$ |
| **45–53** | `wrist_acc_mag_mean`, `wrist_acc_mag_std`, `wrist_acc_mag_max`, `wrist_acc_x_mean`, `wrist_acc_x_std`, `wrist_acc_y_mean`, `wrist_acc_y_std`, `wrist_acc_z_mean`, `wrist_acc_z_std` | Wrist ACC ($32\,\text{Hz}$) | Wrist kinematic displacement norm, motor agitation variance, peak tremor amplitude, and axial orientation | $\text{g}$ |
| **54–64** | `wrist_bvp_mean`, `wrist_bvp_std`, `wrist_bvp_min`, `wrist_bvp_max`, `wrist_bvp_skew`, `wrist_bvp_slope`, `wrist_bvp_peak_count`, `wrist_bvp_hr_bpm`, `wrist_bvp_rr_mean`, `wrist_bvp_rr_std`, `wrist_bvp_rmssd` | Wrist BVP ($64\,\text{Hz}$) | Photoplethysmographic arterial pulse waveform moments, systolic peak detection, pulse rate BPM, and pulse rate variability (PRV) RMSSD | $\text{unitless}, \text{BPM}, \text{s}$ |
| **65–70** | `wrist_eda_mean`, `wrist_eda_std`, `wrist_eda_min`, `wrist_eda_max`, `wrist_eda_skew`, `wrist_eda_slope` | Wrist EDA ($4\,\text{Hz}$) | Palmar/wrist sweat gland tonic conductance, sympathetic burst variance, peak reactivity, and trend slope | $\mu\text{S}$ |
| **71–76** | `wrist_temp_mean`, `wrist_temp_std`, `wrist_temp_min`, `wrist_temp_max`, `wrist_temp_skew`, `wrist_temp_slope` | Wrist Temp ($4\,\text{Hz}$) | Extremity surface temperature, vasomotor dilation/constriction variance, and cutaneous thermal drift | $^\circ\text{C}$ |

---

### 5. Training Methodology and Data Leakage Prevention

#### 5.1 Subject-Independent Validation Strategy (GroupKFold)
A critical pitfall in physiological machine learning is **intra-subject data leakage**. If consecutive sliding windows from the same individual appear in both training and test sets, models simply memorize idiosyncratic subject baselines rather than learning generalized stress patterns.

To prevent this:
- We employ **5-Fold `GroupKFold` Cross-Validation** grouped strictly by `subject_id`.
- In each fold, windows from 12 subjects are used for training, while all windows from the remaining 3 subjects are reserved exclusively for out-of-fold validation.
- **Leakage-Free Feature Scaling:** `StandardScaler` is fitted strictly on the training fold partitions ($X_{\text{train}}$) and applied subsequently to $X_{\text{val}}$.

#### 5.2 Model Hyperparameters (Random Forest Classifier)
- `n_estimators`: $150$ (Ensemble size chosen for low variance and stability)
- `max_depth`: $10$ (Prevents individual trees from memorizing subject-specific high-frequency noise)
- `min_samples_split`: $2$
- `min_samples_leaf`: $1$
- `max_features`: `'sqrt'` ($\sqrt{76} \approx 8$ features considered at each split)
- `bootstrap`: `True`
- `criterion`: `'gini'`
- `random_state`: $42$
- `n_jobs`: $-1$ (Parallelized execution across all CPU cores)

---

### 6. Empirical Evaluation Results and Model Comparisons

#### 6.1 Multi-Algorithm Benchmark Comparison

All models were evaluated using the exact same 5-fold subject-independent out-of-fold cross-validation protocol on the 4,325-sample dataset:

| Model Architecture | Out-of-Fold Accuracy | Precision (Stress) | Recall (Stress) | F1-Score (Stress) | ROC-AUC Score |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression (Standardized)** | $85.11\%$ | $67.59\%$ | $63.23\%$ | $0.6534$ | $0.8057$ |
| **Gradient Boosting (XGBoost)** | $91.42\%$ | $84.20\%$ | $75.52\%$ | $0.7963$ | $0.9628$ |
| **Random Forest Classifier (Selected)** | **$93.66\%$** | **$92.03\%$** | **$78.23\%$** | **$0.8457$** | **$0.9766$** |

#### 6.2 Confusion Matrix (Selected Random Forest Model)

Evaluation on all $N = 4,325$ out-of-fold cross-validation predictions:

```text
                     Predicted Non-Stress (0)    Predicted Stress (1)    Total True
True Non-Stress (0)           3,300                       65               3,365
True Stress (1)                 209                      751                 960
Total Predicted               3,509                      816               4,325
```

- **True Negatives (TN):** $3,300$ ($98.07\%$ specificity on Non-Stress states)
- **False Positives (FP):** $65$ (Extremely low false alarm rate on resting/amusement states)
- **False Negatives (FN):** $209$
- **True Positives (TP):** $751$ ($78.23\%$ sensitivity on acute TSST stress)

#### 6.3 Top-Ranked Physiological Features (Gini Importance)
1. `chest_ecg_rmssd` (Root mean square of successive RR differences — parasympathetic tone indicator)
2. `chest_eda_mean` (Tonic skin conductance level reflecting sympathetic arousal)
3. `chest_ecg_hr_bpm` (Heart rate acceleration during TSST stress)
4. `wrist_eda_mean` (Peripheral sweat gland activation)
5. `chest_temp_slope` (Cutaneous vasoconstrictive cooling trend)
6. `chest_resp_std` (Respiratory dysrhythmia and hyperventilation during stress)
7. `wrist_bvp_rmssd` (Pulse rate variability extracted from optical PPG)

---

### 7. Verification of Subject and Condition Isolation
To guarantee complete technical integrity:
- Every subject's Baseline, Stress, Amusement, and Meditation conditions produce **unique SHA-256 telemetry fingerprints**.
- Baseline and Amusement conditions evaluate to authentic unrounded stress probabilities (e.g., $0.06\%$ or $0.12\%$) without being artificially clamped or identically formatted.
- Meditation conditions evaluate cleanly between $0.05\%$ and $0.95\%$ stress probability, eliminating the legacy 50% fallback artifact.
