# Environmental Context AI Models Evaluation Report

**Generated:** 2026-11-24 18:05:42  
**Status:** EXPERIMENTAL PREDICTION MODELS (TRAINED ON SYNTHETIC DATA)  
**Warning:** SYNTHETIC-DATA TRAINED MODEL — NOT LIVE SENSOR MEASUREMENTS  

## 1. Dataset & Split Specification

- **Dataset:** `BIT_Mesra_Synthetic_Environmental_Dataset.xlsx` / `.csv`
- **Total Samples:** 73,440 records
- **Train Set (Days 1–120):** 48,960 records (Chronological split)
- **Validation Set (Days 121–150):** 12,240 records
- **Test Set (Days 151–180):** 12,240 records
- **Locations:** 17 BIT Mesra campus landmarks
- **Features:** `location`, `hour_of_day`, `day_of_week`, `weekend_indicator`

## 2. Model A: Crowd Density Prediction Comparison

| Model | Val_Accuracy | Val_Macro_Precision | Val_Macro_Recall | Val_Macro_F1 | Test_Accuracy | Test_Macro_Precision | Test_Macro_Recall | Test_Macro_F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Dummy Classifier | 0.4858 | 0.1619 | 0.3333 | 0.218 | 0.4806 | 0.1602 | 0.3333 | 0.2164 |
| Logistic Regression (Balanced) | 0.4464 | 0.4315 | 0.4368 | 0.4184 | 0.4422 | 0.4245 | 0.4294 | 0.4125 |
| Random Forest (Balanced) | 0.719 | 0.6513 | 0.6534 | 0.6468 | 0.7191 | 0.6566 | 0.6545 | 0.6499 |
| Gradient Boosting (XGBoost Balanced) | 0.7126 | 0.6668 | 0.6758 | 0.6706 | 0.7132 | 0.6693 | 0.6761 | 0.6722 |

**Selected Model:** `Gradient Boosting (XGBoost Balanced)`  
**Test Confusion Matrix (Rows=True, Cols=Pred: Low, Medium, High):**

```
[[3112  293  296]
 [ 715 1126  815]
 [ 141 1250 4492]]
```

### Per-Class Test Performance (Model A)

| index | precision | recall | f1-score | support |
| --- | --- | --- | --- | --- |
| Low | 0.7842741935483871 | 0.8408538232910024 | 0.811579084626418 | 3701.0 |
| Medium | 0.421880854252529 | 0.4239457831325301 | 0.4229107981220657 | 2656.0 |
| High | 0.8017133678386579 | 0.7635560088390277 | 0.7821695977711998 | 5883.0 |
| accuracy | 0.7132352941176471 | 0.7132352941176471 | 0.7132352941176471 | 0.7132352941176471 |
| macro avg | 0.669289471879858 | 0.6761185384208535 | 0.6722198268398945 | 12240.0 |
| weighted avg | 0.7140191243637355 | 0.7132352941176471 | 0.7131053117404043 | 12240.0 |

### 2.1 Crowd Classification — Medium Class Performance & Generator Rule Investigation

#### Root Cause Analysis
An in-depth inspection of the synthetic dataset generation pipeline (`src/environment/generate_synthetic_dataset.py`) revealed why standard unweighted classifiers exhibited poor recall on the `Medium` crowd density class (~4.48%, identifying only 119 out of 2,656 test records):
1. **Absence of Modal Target Status:** Across all 17 BIT Mesra campus landmarks, 24 diurnal hours, and weekday/weekend schedules, the synthetic generation rules assign either `High` (e.g. academic peak hours, hostel evenings) or `Low` (e.g. late night, remote zones) as the baseline modal target.
2. **Transition Perturbation Artifact:** The `Medium` category was generated solely as a transition perturbation with probabilities $P \le 0.42$ (e.g., $10\%$, $15\%$, $30\%$, or $42\%$). Consequently, for any unique feature vector $\mathbf{x} = (\text{location}, \text{hour}, \text{dow}, \text{weekend})$, `Medium` was never the majority class.
3. **Loss Minimization Suppression:** Standard gradient boosting and tree classifiers minimizing unweighted cross-entropy loss naturally learn to predict the conditional mode (`High` or `Low`), effectively suppressing the minority transition class.

#### Scientifically Justified Correction: Cost-Sensitive Balanced Sample Weighting
Rather than modifying the test set or fabricating artificial records (which would compromise academic integrity), we implemented cost-sensitive balanced sample weighting:
$$w_c = \frac{N}{C \cdot N_c}$$
computed on the training split using `sklearn.utils.class_weight.compute_sample_weight('balanced', y_train)`.

#### Comparative Evaluation Results:
- **Unweighted Baseline XGBoost:**
  - Test Macro F1: `0.5702`
  - Medium Recall: `4.48%` (119 / 2,656 correct)
  - Validation Macro F1: `0.5782`
- **Balanced Sample-Weighted XGBoost (Selected Model):**
  - Test Macro F1: `0.6722` (+10.2 percentage points)
  - Medium Recall: `42.39%` (1,126 / 2,656 correct, **~9.5× improvement**)
  - Test Accuracy: `71.32%` (maintained high overall accuracy)
  - Validation Macro F1: `0.6706` (consistent with test performance, verifying genuine stability without overfitting)

## 3. Model B: Lighting Condition Prediction Comparison

| Model | Val_Accuracy | Val_Macro_Precision | Val_Macro_Recall | Val_Macro_F1 | Test_Accuracy | Test_Macro_Precision | Test_Macro_Recall | Test_Macro_F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Dummy Classifier | 0.7115 | 0.2372 | 0.3333 | 0.2771 | 0.7095 | 0.2365 | 0.3333 | 0.2767 |
| Logistic Regression | 0.7353 | 0.771 | 0.4031 | 0.3943 | 0.7342 | 0.8271 | 0.4037 | 0.3957 |
| Random Forest | 0.8895 | 0.8357 | 0.765 | 0.795 | 0.8846 | 0.8265 | 0.7527 | 0.7833 |
| Gradient Boosting (XGBoost) | 0.8895 | 0.8357 | 0.765 | 0.795 | 0.8846 | 0.8265 | 0.7527 | 0.7833 |

**Selected Model:** `Random Forest`  
**Test Confusion Matrix (Rows=True, Cols=Pred: Dark, Low, Good):**

```
[[ 915  460   92]
 [ 142 1356  591]
 [  23  104 8557]]
```

### Per-Class Test Performance (Model B)

| index | precision | recall | f1-score | support |
| --- | --- | --- | --- | --- |
| Dark | 0.8472222222222222 | 0.623721881390593 | 0.71849234393404 | 1467.0 |
| Low | 0.70625 | 0.6491144088080422 | 0.6764779246694936 | 2089.0 |
| Good | 0.9260822510822511 | 0.9853754030400736 | 0.9548091943762553 | 8684.0 |
| accuracy | 0.884640522875817 | 0.884640522875817 | 0.884640522875817 | 0.884640522875817 |
| macro avg | 0.8265181577681577 | 0.7527372310795696 | 0.783259820993263 | 12240.0 |
| weighted avg | 0.8791118887580285 | 0.884640522875817 | 0.8789831451919289 | 12240.0 |

## 4. Experimental Continuous Lux Estimation (Regression)

- **Model:** Random Forest Regressor (100 trees, max_depth=10)
- **Test RMSE:** 650.54 lux
- **Test MAE:** 378.1 lux
- **Test R² Score:** 0.7571
*(Note: Lux target is synthetically simulated and must not be interpreted as physical photometer measurements)*

## 5. Methodological Limitations & Transparency

1. **Direct Generator Rule Encoding:** Because the synthetic dataset was generated using campus baseline schedule rules, high model scores indicate that the algorithms have effectively learned the deterministic rule mappings rather than empirical physical reality.
2. **Absence of Real-time Optical/Acoustic Sensors:** The model predicts expected schedule conditions based on campus geography and calendar clock time. It cannot detect dynamic impromptu gatherings, power grid blackouts, or adverse weather conditions.
