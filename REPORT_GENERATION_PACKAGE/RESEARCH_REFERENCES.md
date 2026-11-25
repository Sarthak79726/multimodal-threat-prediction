# Multimodal Threat Prediction Engine — Version 1.0
## Technical Documentation: Academic Research References, Citations, and Bibliographic Foundation

---

### 1. Primary Empirical Dataset References

#### [1] The WESAD Benchmark Dataset
- **Full Citation:** Schmidt, Philip; Reiss, Attila; Duerichen, Robert; Marberger, Claus; Van Laerhoven, Kristof. (2018). *"Introducing WESAD, a Multimodal Dataset for Wearable Stress and Affect Detection."* In *Proceedings of the 20th ACM International Conference on Multimodal Interaction (ICMI '18)*, Boulder, CO, USA, pp. 400–408.
- **DOI:** [10.1145/3242969.3242985](https://doi.org/10.1145/3242969.3242985)
- **Relevance to Project:** Serves as the primary empirical foundation for the physiological distress branch (Branch 1). Provides synchronized multi-sensor recordings (chest RespiBAN and wrist Empatica E4) across 15 human subjects subjected to the Trier Social Stress Test (TSST), amusement videos, and guided meditation.
- **Verification Status:** Fully verified in repository (`data/WESAD/`, `src/physiological/`).

#### [2] The Trier Social Stress Test (TSST)
- **Full Citation:** Kirschbaum, Clemens; Pirke, Karl-Martin; Hellhammer, Dirk H. (1993). *"The 'Trier Social Stress Test' – A Tool for Investigating Psychobiological Stress Responses in a Laboratory Setting."* *Neuropsychobiology*, 28(1–2), pp. 76–81.
- **DOI:** [10.1159/000119004](https://doi.org/10.1159/000119004)
- **Relevance to Project:** The empirical laboratory protocol utilized in WESAD to induce acute social-evaluative threat and sympathetic nervous system activation, defining the ground-truth "Stress" class (Label 2).
- **Verification Status:** Fully verified.

---

### 2. Machine Learning and Algorithm Foundations

#### [3] Random Forest Classifier
- **Full Citation:** Breiman, Leo. (2001). *"Random Forests."* *Machine Learning*, 45(1), pp. 5–32.
- **DOI:** [10.1023/A:1010933404324](https://doi.org/10.1023/A:1010933404324)
- **Relevance to Project:** Core algorithm selected for the physiological stress detector (150 estimators, out-of-fold ROC-AUC = $0.9766$), the environmental lighting classifier (100 estimators, test accuracy = $88.46\%$), and the continuous lux regressor ($R^2 = 0.7571$).
- **Verification Status:** Fully verified in `models/physiological/` and `models/environment/`.

#### [4] Extreme Gradient Boosting (XGBoost)
- **Full Citation:** Chen, Tianqi; Guestrin, Carlos. (2016). *"XGBoost: A Scalable Tree Boosting System."* In *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD '16)*, San Francisco, CA, USA, pp. 785–794.
- **DOI:** [10.1145/2939672.2939785](https://doi.org/10.1145/2939672.2939785)
- **Relevance to Project:** Core algorithm deployed for the crowd density prediction model (`Gradient Boosting (XGBoost Balanced)`). Implements cost-sensitive sample weighting ($w_c = \frac{N}{C \cdot N_c}$), lifting transitional `Medium` crowd recall from $4.48\%$ to $42.39\%$.
- **Verification Status:** Fully verified in `models/environment/crowd_model.pkl`.

#### [5] Stacked Generalization (Late Fusion Architecture)
- **Full Citation:** Wolpert, David H. (1992). *"Stacked Generalization."* *Neural Networks*, 5(2), pp. 241–259.
- **DOI:** [10.1016/S0893-6080(05)80023-1](https://doi.org/10.1016/S0893-6080(05)80023-1)
- **Relevance to Project:** Foundational literature underpinning multi-branch late fusion, level-0 base model featurization, and level-1 meta-learner stacking investigated in `src/fusion/stacking_pipeline.py`.
- **Verification Status:** Verified.

---

### 3. Biometric Signal Processing and HRV Analytics

#### [6] Heart Rate Variability (HRV) Metrics and Norms
- **Full Citation:** Shaffer, Fred; Ginsberg, J. P. (2017). *"An Overview of Heart Rate Variability Metrics and Norms."* *Frontiers in Public Health*, 5, Article 258, pp. 1–17.
- **DOI:** [10.3389/fpubh.2017.00258](https://doi.org/10.3389/fpubh.2017.00258)
- **Relevance to Project:** Guides the physiological feature extraction pipeline (`src/physiological/feature_extraction.py`), specifically the computation of time-domain inter-beat interval metrics: mean RR interval, SDNN (`chest_ecg_rr_std`), and RMSSD (`chest_ecg_rmssd`).
- **Verification Status:** Fully verified.

#### [7] Electrodermal Activity (EDA) Standards
- **Full Citation:** Boucsein, Wolfram; Fowles, Don C.; Grimnes, Sverre; Ben-Shakhar, Gershon; Roth, Walton T.; Dawson, Michael E.; Filion, Diane L. (2012). *"Publication Recommendations for Electrodermal Measurements."* *Psychophysiology*, 49(8), pp. 1017–1034.
- **DOI:** [10.1111/j.1469-8986.2012.01384.x](https://doi.org/10.1111/j.1469-8986.2012.01384.x)
- **Relevance to Project:** Informs the extraction of tonic and phasic electrodermal features from chest RespiBAN and wrist Empatica E4 signals.
- **Verification Status:** Fully verified.

---

### 4. Geospatial, Mathematical, and Calibration Literature

#### [8] Haversine Great-Circle Distance
- **Full Citation:** Sinnott, Roger W. (1984). *"Virtues of the Haversine."* *Sky and Telescope*, 68(2), p. 159.
- **Relevance to Project:** Underpins the spherical trigonometric distance formulation in `src/location/campus_map.py` (`haversine_distance()`), calculating meter distances across the Earth's curvature without planar map projection distortion.
- **Verification Status:** Fully verified.

#### [9] Probability Calibration in Supervised Learning
- **Full Citation:** Niculescu-Mizil, Alexandru; Caruana, Rich. (2005). *"Predicting Good Probabilities with Supervised Learning."* In *Proceedings of the 22nd International Conference on Machine Learning (ICML '05)*, Bonn, Germany, pp. 625–632.
- **DOI:** [10.1145/1102351.1102430](https://doi.org/10.1145/1102351.1102430)
- **Relevance to Project:** Informs the Platt scaling and isotonic regression modules in `src/fusion/calibration.py`, ensuring multi-branch threat probabilities reflect true empirical likelihoods.
- **Verification Status:** Fully verified.

#### [10] OpenStreetMap (OSM) BIT Mesra Geographic Survey
- **Full Citation:** OpenStreetMap Contributors. (2025). *"Birla Institute of Technology, Mesra Campus Geographic Survey and Spatial Cartography."* Planet OSM.
- **URL:** [https://www.openstreetmap.org](https://www.openstreetmap.org)
- **Relevance to Project:** Source cartographic data for the 17 verified campus landmark coordinates, detection radii, and perimeter boundaries in `src/location/landmarks.py`.
- **Verification Status:** Verified.
