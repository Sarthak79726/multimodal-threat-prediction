# ⚠️ Scientific Limitations and Ethical Considerations

**Project:** Multimodal Threat Prediction Engine (B.Tech AIML Project)  
**Institution:** Birla Institute of Technology (BIT) Mesra  
**Document Purpose:** Mandatory Academic Transparency and Ethical Safety Disclosures

---

## 1. Scientific Limitations

### Limitation 1: Absence of Synchronized Empirical Multimodal Threat Data
The project currently does not possess a synchronized empirical dataset combining wearable physiological telemetry, client GPS coordinates, and ambient environmental logs collected during genuine physical security incidents. Consequently, a production-grade Level 1 Stacking Meta-Learner **cannot be legitimately trained at this stage**. The production meta-learner status is therefore marked as:
`UNTRAINED — VALID MULTIMODAL THREAT DATASET REQUIRED`.

### Limitation 2: Physiological Modality Semantics (Stress vs. Threat)
The physiological classification engine is trained on the **WESAD** (Wearable Stress and Affect Detection) benchmark. 
- WESAD induces **laboratory stress** via the Trier Social Stress Test (mental arithmetic, public speaking).
- **Stress is not threat:** Sympathetic nervous system arousal (elevated heart rate, galvanic skin response, skin temperature drop) occurs during cognitive exertion, physical exercise, excitement, caffeine consumption, and acute stress.
- Renaming stress detection as "threat detection" is scientifically invalid. Physiological arousal is an associative indicator, not a definitive sensor of physical assault or danger.

### Limitation 3: Spatial Engine Nature (Deterministic vs. Statistical)
The campus spatial engine (`src/location/model.py`) is a **deterministic geodesic calculator**, not an empirical crime prediction model.
- It computes great-circle Haversine distances to 17 BIT Mesra campus landmarks and security posts.
- Parameters such as zone lighting ratings and spatial isolation indices are cataloged static heuristics, not statistically fitted regression coefficients.
- The engine does not account for micro-topography, building interiors, dynamic foot traffic, or localized security patrol schedules.

### Limitation 4: Environmental Engine Nature (Rule-Based Heuristic)
The environmental branch (`src/environment/model.py`) is an **expert rule-based evaluator**.
- Scores are computed from fixed additive heuristic weights assigned to diurnal hours, binary illumination (`Light`/`Dark`), and categorical crowd density (`Low`/`Medium`/`High`).
- The weights are intuitive approximations rather than empirically estimated log-odds from historical safety incident data.

### Limitation 5: Active Inference Nature (Heuristic Index vs. Calibrated Probability)
Because the Level 1 Meta-Learner is untrained, the active score displayed by the system is the **Experimental Weighted Context Index**:
$$\text{Index} = 0.40 \cdot P_{\text{physio}} + 0.25 \cdot P_{\text{spatial}} + 0.35 \cdot P_{\text{env}}$$
- This calculation is an un-trained weighted reference benchmark.
- **It is a heuristic index, NOT a calibrated probability of physical threat.**
- It must never be presented as an empirical risk probability or confidence interval.

---

## 2. Ethical and Life-Safety Considerations

### Life Safety Warning
> [!CAUTION]
> **NOT A LIFE-SAFETY SYSTEM:** The Multimodal Threat Prediction Engine is an academic research prototype and proof-of-concept. It must **NEVER** be relied upon as an autonomous personal safety protector, an emergency dispatch trigger, or a replacement for campus security personnel, emergency telephones, or law enforcement contact.

### The Ethics of Classification Errors in Personal Safety
1. **Consequences of False Negatives (FN):** A false negative occurs when a user faces actual physical danger but the system outputs a "STANDARD CONTEXT" assessment. If a user relies on the software for personal safety reassurance, an undetected threat could result in severe physical harm.
2. **Consequences of False Positives (FP):** Repeated false alarms caused by physiological stress (e.g., hurrying between lecture halls, exam anxiety) induce alert fatigue, panic, and loss of user trust, leading users to disable or ignore warnings.

### Data Privacy and Continuous Surveillance Risks
1. **Biometric Privacy:** Physiological signals (ECG, BVP) contain sensitive personal health markers (cardiac arrhythmias, respiratory conditions, emotional states). Continuous transmission or storage of raw biosignals presents significant privacy vulnerabilities under data protection standards.
2. **Geospatial Tracking:** Continuous client GPS tracking exposes individual movement trajectories, daily routines, residential addresses, and personal habits. Any production system must incorporate on-device anonymization, local processing, and ephemeral data retention.

---

## 3. Academic Integrity Statement

In accordance with ethical academic research standards:
- All synthetic demonstration models and simulation benchmarks are strictly isolated in `models/fusion/synthetic_demo/` and `reports/synthetic_simulation/`.
- Synthetic data is never presented as empirical ground truth.
- Synthetic simulation metrics (ROC-AUC 0.9939, Brier 0.0181) are explicitly documented as mathematical code verification benchmarks.
- The production user interface transparently communicates the untrained status of the meta-learner and the heuristic nature of the active context index.
