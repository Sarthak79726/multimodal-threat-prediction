# Environmental Context AI Models Evaluation Report

**Generated:** 2026-11-24 16:16:34  
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
| Logistic Regression | 0.5978 | 0.3961 | 0.4893 | 0.4355 | 0.5964 | 0.3946 | 0.4884 | 0.4339 |
| Random Forest | 0.7371 | 0.6482 | 0.6201 | 0.558 | 0.7272 | 0.6314 | 0.6118 | 0.5501 |
| Gradient Boosting (XGBoost) | 0.737 | 0.6515 | 0.6264 | 0.5782 | 0.7267 | 0.6436 | 0.6176 | 0.5702 |

**Selected Model:** `Gradient Boosting (XGBoost)`  
**Test Confusion Matrix (Rows=True, Cols=Pred: Low, Medium, High):**

```
[[3156   37  508]
 [ 733  119 1804]
 [ 149  114 5620]]
```

### Per-Class Test Performance (Model A)

| index | precision | recall | f1-score | support |
| --- | --- | --- | --- | --- |
| Low | 0.7815750371471025 | 0.8527425020264793 | 0.8156092518413232 | 3701.0 |
| Medium | 0.44074074074074077 | 0.04480421686746988 | 0.08133971291866028 | 2656.0 |
| High | 0.708522440746344 | 0.9552949175590685 | 0.813608396670286 | 5883.0 |
| accuracy | 0.7267156862745098 | 0.7267156862745098 | 0.7267156862745098 | 0.7267156862745098 |
| macro avg | 0.6436127395447291 | 0.6176138788176725 | 0.5701857871434232 | 12240.0 |
| weighted avg | 0.6725044231045404 | 0.7267156862745098 | 0.6553158755055548 | 12240.0 |

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
