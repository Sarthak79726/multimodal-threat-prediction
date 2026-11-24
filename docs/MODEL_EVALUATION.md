# 📊 Model Evaluation and Benchmarking Framework

**Project:** Multimodal Threat Prediction Engine  
**Topic:** Evaluation Metrics, Probabilistic Calibration, and Simulation vs. Empirical Benchmarking  
**Important Notice:** All high-performance stacking metrics reported in initial experiments (ROC-AUC 0.9939, Brier 0.0181) originate from an **isolated synthetic simulation demonstration** and must NOT be interpreted as empirical real-world threat prediction performance.

---

## 1. Evaluation Metric Suite

Personal safety and threat prediction systems require evaluation across both **classification discrimination** and **probabilistic calibration**. A model that discriminates well but assigns uncalibrated probabilities (e.g., claiming 95% risk when actual risk is 10%) can cause panic and alert fatigue.

### A. Discrimination Metrics
- **ROC-AUC (Area Under ROC Curve):** Measures separation across all decision thresholds. Insensitive to class imbalance, providing a broad assessment of ranking ability.
- **PR-AUC (Precision-Recall AUC):** Highly informative in rare-event settings (e.g., personal safety incidents), quantifying the precision-recall trade-off without being distorted by large true-negative counts.
- **F1-Score:** Harmonic mean of Precision and Recall at the operational threshold ($\tau = 0.50$).
- **Recall (Sensitivity):** $\frac{\text{TP}}{\text{TP} + \text{FN}}$. Quantifies the proportion of actual safety threats correctly detected. In safety-critical systems, high recall is vital to avoid fatal false negatives.
- **Specificity:** $\frac{\text{TN}}{\text{TN} + \text{FP}}$. Quantifies the proportion of safe situations correctly identified, minimizing false alarms.

### B. Probabilistic Calibration Metrics
- **Brier Score:** Mean squared error between predicted probabilities $\hat{p}_k$ and binary outcomes $y_k \in \{0, 1\}$:
  $$\text{Brier} = \frac{1}{N} \sum_{k=1}^N (\hat{p}_k - y_k)^2$$
  A lower Brier score indicates sharper, better-calibrated probabilities.
- **Expected Calibration Error (ECE):** Partitions predicted probabilities into $M$ equal-width bins $B_m$ and computes the weighted absolute difference between bin accuracy and bin confidence:
  $$\text{ECE} = \sum_{m=1}^M \frac{|B_m|}{N} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$
  An ECE close to 0 indicates that when the model predicts $p\%$, the true outcome frequency in that bin is indeed approximately $p\%$.

---

## 2. Benchmark Architectures Compared

Our benchmarking framework in `src/fusion/evaluation.py` evaluates seven candidate configurations:

1. **Physiological Branch Only:** Single-modality stress classifier.
2. **Spatial Branch Only:** Single-modality geodesic campus risk score.
3. **Environmental Branch Only:** Single-modality rule-based situational score.
4. **Weighted Late Fusion Baseline:** Un-trained linear combination ($0.40 \cdot P_1 + 0.25 \cdot P_2 + 0.35 \cdot P_3$).
5. **Stacking (Logistic Regression):** Level 1 linear meta-learner with L2 regularization.
6. **Stacking (Random Forest):** Level 1 non-linear tree ensemble meta-learner.
7. **Stacking (XGBoost):** Level 1 gradient-boosted decision tree meta-learner.

---

## 3. Synthetic Simulation Demonstration Results

To verify the mathematical pipeline mechanics, out-of-fold matrix generation, and calibration routines, a controlled synthetic simulation was executed:
- **Sample Count:** 4,000 synthetic observations (3,189 train / 811 untouched test).
- **Subject Identifiers:** 20 artificial tokens (`Subj_01` through `Subj_20`).
- **Feature Distribution:** Drawn from Beta distributions with non-linear interaction terms.
- **Label Generation:** Non-linear mathematical hazard formula.

### Simulation Benchmark Table (Untouched Test Partition, $N=811$):

| Architecture | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | Brier | ECE |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1. Physiological Only | 0.8286 | 0.2105 | 0.9000 | 0.3412 | 0.9235 | 0.4222 | 0.1257 | 0.2663 |
| 2. Spatial Only | 0.7041 | 0.1241 | 0.8250 | 0.2157 | 0.8414 | 0.1927 | 0.1891 | 0.3449 |
| 3. Environmental Only | 0.8397 | 0.0982 | 0.2750 | 0.1447 | 0.6325 | 0.0869 | 0.1214 | 0.2107 |
| **4. Late Fusion Baseline** | 0.9630 | 0.5926 | 0.8000 | 0.6809 | 0.9803 | 0.7364 | 0.1086 | 0.2718 |
| **5. Stacking (LogReg Demo)**| **0.9741** | **1.0000** | 0.4750 | 0.6441 | **0.9939** | **0.9009** | **0.0181** | **0.0301** |
| 6. Stacking (RF Demo) | 0.9692 | 0.8947 | 0.4250 | 0.5763 | 0.9823 | 0.8196 | 0.0230 | 0.0277 |
| 7. Stacking (XGBoost Demo) | 0.9790 | 0.8286 | 0.7250 | 0.7733 | 0.9902 | 0.8933 | 0.0148 | 0.0082 |

### Methodological Takeaways from Simulation:
1. **Mathematical Pipeline Functionality:** Demonstrates that the `OOFGenerator` and `StackingEnsemblePipeline` successfully eliminate data leakage and optimize log-loss parameters without numeric errors.
2. **Calibration Superiority of Stacking:** Demonstrates that stacking reduces calibration error by over 9× compared to raw weighted sums (Brier: 0.0181 vs 0.1086; ECE: 0.0301 vs 0.2718).
3. **Strict Academic Isolation:** Because the underlying data was generated via NumPy distributions, these metrics represent **simulation benchmark results** and must never be cited as real-world threat detection accuracy.

---

## 4. Empirical Model Deployment Checklist

Before any future model can be deployed into production or marked `production_eligible = True`, it must satisfy every item on this checklist:

- [ ] Training data collected from at least 30 independent human subjects.
- [ ] Multimodal signals synchronized to UTC millisecond precision ($\le 250$ms drift).
- [ ] Ground-truth labels sourced from independently verified physical security incident reports or verified drill logs.
- [ ] Dataset verified by `StackingEnsemblePipeline.validate_multimodal_dataset(..., enforce_production_empirical=True)`.
- [ ] Evaluation performed on an untouched holdout set with zero subject overlap.
- [ ] Brier score and ECE evaluated and verified to outperform the baseline.
- [ ] Provenance metadata serialised within the artifact header.
