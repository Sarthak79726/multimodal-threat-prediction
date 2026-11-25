# Multimodal Threat Prediction Engine — Version 1.0
## Technical Documentation: Multimodal Late Fusion, Missing-Data Renormalization, and Scoring Formulations

---

### 1. Conceptual Framework of Multimodal Fusion
A central thesis of this project is that **no single sensing modality is sufficient to reliably detect physical threat or acute distress**:
- **Physiological biometric monitoring alone** cannot distinguish between malevolent distress (assault, stalking) and benign physiological arousal (vigorous exercise, social excitement, intense amusement, or academic exam anxiety).
- **Spatial location alone** creates constant false alerts if an individual is simply walking peacefully through a secluded campus sector at night.
- **Environmental schedules alone** indicate contextual probability of darkness or isolation, but cannot detect whether an individual is in actual danger.

By synthesizing these three orthogonal axes into a unified context index, the system suppresses false alarms during benign arousal in safe zones while heightening sensitivity in hazardous locations.

---

### 2. Modality Output Range and Preprocessing

| Input Modality Branch | Mathematical Symbol | Domain / Range | Physical Interpretation | Preprocessing & Boundary Guard |
| :--- | :---: | :---: | :--- | :--- |
| **Physiological Stress** | $P$ | $[0.0, 1.0]$ | Calibrated Random Forest probability of acute somatic stress (WESAD TSST) | Clipped to $[0.0, 1.0]$; non-finite values default to median $0.18$ with warning |
| **Spatial Vulnerability** | $L$ | $[0.0, 1.0]$ | Campus geospatial isolation, security distance, and time-adjusted lighting | Clipped to $[0.0, 1.0]$; non-finite values default to baseline $0.15$ with warning |
| **Environmental Hazard**| $E$ | $[0.05, 1.00]$| Composite score reflecting predicted darkness, low crowd, and nocturnal schedule | Strictly validated; if `None` or non-finite, triggers missing-data branch |

---

### 3. Mathematical Formulations of Multimodal Fusion

#### 3.1 Complete Three-Branch Weighted Late Fusion
When all three sensing modalities are active and report valid data, the primary operational threat index $R \in [0.0, 1.0]$ is computed as:

$$R = w_P \cdot P + w_L \cdot L + w_E \cdot E$$

Under the standard operational baseline weights established for BIT Mesra:
- **Physiological Weight ($w_P$):** $0.40$ ($40\%$ decision importance)
- **Spatial Context Weight ($w_L$):** $0.25$ ($25\%$ decision importance)
- **Environmental Context Weight ($w_E$):** $0.35$ ($35\%$ decision importance)

$$\sum w_i = 0.40 + 0.25 + 0.35 = 1.00$$

$$R = 0.40 \cdot P + 0.25 \cdot L + 0.35 \cdot E$$

#### 3.2 The Missing-Data Challenge: Rejection of Silent Imputation
In earlier prototypes and naive architectures, missing sensor streams (such as an uninitialized environmental model or sensor disconnection) were frequently imputed with an arbitrary constant (e.g., $E = 0.20$).

**The Scientific Flaw:** Imputing $E = 0.20$ artificially depresses high-threat situations (reducing a true $85\%$ threat to $67\%$) or elevates low-threat situations without operator awareness.

#### 3.3 The Scientifically Justified Correction: Weight Renormalization
When the Environmental branch is unavailable ($E = \text{None}$ or uninitialized), the system:
1. Disables the 3-branch index: `experimental_context_index = None`.
2. Emits an explicit system banner:
   > *"Complete 3-branch index is unavailable. The engine evaluates a Partial Dual-Branch Context Index (Physiological + Spatial only). No neutral values are silently substituted."*
3. Evaluates a mathematically rigorous **Partial Dual-Branch Context Index** by renormalizing the remaining active modality weights:

$$w_P' = \frac{w_P}{w_P + w_L} = \frac{0.40}{0.40 + 0.25} = \frac{0.40}{0.65} \approx 0.615385$$

$$w_L' = \frac{w_L}{w_P + w_L} = \frac{0.25}{0.40 + 0.25} = \frac{0.25}{0.65} \approx 0.384615$$

$$R_{\text{dual}} = \frac{0.40 \cdot P + 0.25 \cdot L}{0.65}$$

Notice that $w_P' + w_L' = 1.000$, preserving scale invariance across $[0.0, 1.0]$ without artificial bias.

---

### 4. Decision Thresholds and Threat Categorization

The continuous context index ($R$ or $R_{\text{dual}}$) is mapped into three standardized operational threat levels:

$$\text{Threat Level} = \begin{cases}
\text{LOW RISK (Safe / Normal Operation)} & \text{if } R \le 0.39 \\
\text{MEDIUM RISK (Elevated Caution / Monitoring Required)} & \text{if } 0.40 \le R \le 0.69 \\
\text{HIGH RISK (Critical Alarm / Immediate Operator Verification)} & \text{if } R \ge 0.70
\end{cases}$$

---

### 5. Grounded Explainability: Modality Contribution Breakdown
To support transparent human-in-the-loop decision-making, the dashboard decomposes the overall score into exact percentage contributions:

$$\text{Contribution}_P = \frac{w_P \cdot P}{R} \times 100\%$$

$$\text{Contribution}_L = \frac{w_L \cdot L}{R} \times 100\%$$

$$\text{Contribution}_E = \frac{w_E \cdot E}{R} \times 100\%$$

This allows emergency response personnel to immediately distinguish whether an alert is driven primarily by acute physiological distress ($> 60\%$) or severe geographic isolation.

---

### 6. Five Worked Illustrative Numerical Examples

#### Example 1: Critical Threat Scenario (Acute Distress in Perimeter Forest at Night)
- **Scenario:** A student experiencing severe distress near Hostel 13 at 23:30.
- **Branch Inputs:**
  - Physiological: $P = 0.9500$ (Acute TSST stress response, elevated HR and skin conductance)
  - Spatial: $L = 0.8500$ (Hostel 13 perimeter boundary, $600\,\text{m}$ from security post, night penalty)
  - Environmental: $E = 0.7200$ (Nocturnal hour, predicted `Dark` lighting, predicted `Low` crowd)
- **Calculation:**
  $$R = 0.40(0.9500) + 0.25(0.8500) + 0.35(0.7200)$$
  $$R = 0.3800 + 0.2125 + 0.2520 = \mathbf{0.8445}$$
- **Threat Level:** **HIGH RISK ($84\%$)**
- **Contribution Breakdown:**
  - Physiological: $\frac{0.3800}{0.8445} \times 100\% = 45.0\%$
  - Spatial: $\frac{0.2125}{0.8445} \times 100\% = 25.2\%$
  - Environmental: $\frac{0.2520}{0.8445} \times 100\% = 29.8\%$

---

#### Example 2: Normal Baseline Scenario (Resting Student in Main Building During Day)
- **Scenario:** A calm student studying in the Main Building administrative sector at 14:00.
- **Branch Inputs:**
  - Physiological: $P = 0.0500$ (Resting baseline telemetry)
  - Spatial: $L = 0.0800$ (Main Building, $20\,\text{m}$ from 24/7 security checkpoint, daytime)
  - Environmental: $E = 0.1200$ (Daytime sunlight $> 2000\,\text{lx}$, predicted `High` crowd)
- **Calculation:**
  $$R = 0.40(0.0500) + 0.25(0.0800) + 0.35(0.1200)$$
  $$R = 0.0200 + 0.0200 + 0.0420 = \mathbf{0.0820}$$
- **Threat Level:** **LOW RISK ($8\%$)**

---

#### Example 3: Missing Environmental Sensor Data (Renormalized Dual-Branch Evaluation)
- **Scenario:** Student at Hostel 6 in the evening; environmental sensor stream is disconnected.
- **Branch Inputs:**
  - Physiological: $P = 0.8000$ (Elevated physiological stress)
  - Spatial: $L = 0.5000$ (Hostel 6 sector in evening)
  - Environmental: $E = \text{None}$ (`UNAVAILABLE`)
- **Calculation (Dual-Branch Fallback):**
  $$R_{\text{dual}} = \frac{0.40(0.8000) + 0.25(0.5000)}{0.65} = \frac{0.3200 + 0.1250}{0.65} = \frac{0.4450}{0.65} \approx \mathbf{0.6846}$$
- **Threat Level:** **MEDIUM RISK ($68\%$)**
- **Status:** Evaluated as `PARTIAL_DUAL_BRANCH`. Complete 3-branch index marked unavailable without silent neutral padding.

---

#### Example 4: Non-Threatening High Arousal (Benign Amusement in Safe Academic Zone)
- **Scenario:** Student watching a comedy video or experiencing excitement at the R&D center at 15:00.
- **Branch Inputs:**
  - Physiological: $P = 0.7500$ (Elevated sympathetic arousal, laughter heart rate spike)
  - Spatial: $L = 0.1200$ (R&D academic zone, well-lit, close to library staff)
  - Environmental: $E = 0.1200$ (Daytime ambient illumination, moderate foot traffic)
- **Calculation:**
  $$R = 0.40(0.7500) + 0.25(0.1200) + 0.35(0.1200)$$
  $$R = 0.3000 + 0.0300 + 0.0420 = \mathbf{0.3720}$$
- **Threat Level:** **LOW RISK ($37\%$)**
- **Key Insight:** A standalone physiological monitor would trigger a false alarm ($75\% \ge 70\%$). Multimodal fusion correctly pulls the composite score below the alarm threshold ($37\% \le 39\%$), demonstrating robust false-positive suppression.

---

#### Example 5: Solitary Late-Night Walk Without Distress (Situational Vigilance)
- **Scenario:** A calm student walking past Hostel 12 perimeter at 01:00 AM.
- **Branch Inputs:**
  - Physiological: $P = 0.0600$ (Completely calm, normal breathing and pulse)
  - Spatial: $L = 0.8500$ (Hostel 12 perimeter, high night isolation)
  - Environmental: $E = 0.7200$ (Deep night, dark forest border)
- **Calculation:**
  $$R = 0.40(0.0600) + 0.25(0.8500) + 0.35(0.7200)$$
  $$R = 0.0240 + 0.2125 + 0.2520 = \mathbf{0.4885}$$
- **Threat Level:** **MEDIUM RISK ($49\%$)**
- **Key Insight:** The system does not raise a critical red alarm because biometrics are calm, but correctly identifies elevated environmental vulnerability, prompting passive monitoring without panic dispatch.

---

### 7. Status of Production Stacking and Dynamic Gating Models

#### Production Stacking Meta-Learner Status
- In `src/fusion/stacking_pipeline.py`, a Level 1 Stacking Ensemble pipeline was prototyped.
- **Academic Governance Decision:** The stacking meta-learner is explicitly flagged as `is_trained = False` and gated by `IneligibleProductionArtifactError`.
- **Reason:** Training a supervised second-level meta-learner requires synchronized ground-truth physical assault labels paired with simultaneous WESAD signals and GPS tracks. Because no genuine physical assault dataset exists, training a meta-learner on synthetic demo labels would constitute pseudo-scientific fabrication.
- **Version 1.0 Policy:** The operational system strictly employs the transparent Weighted Late Fusion baseline.
