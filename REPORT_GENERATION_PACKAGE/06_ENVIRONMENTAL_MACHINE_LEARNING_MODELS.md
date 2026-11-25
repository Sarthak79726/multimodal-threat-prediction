# Multimodal Threat Prediction Engine — Version 1.0
## Technical Documentation: Environmental Machine Learning Models and Cost-Sensitive Classification

---

### 1. Executive Summary of Environmental Models
The Environmental Context branch deploys three dedicated machine learning architectures trained on the 180-day diurnal simulation dataset ($N = 73,440$ rows, chronological train/val/test splits):
1. **Model A (Crowd Density Classifier):** Gradient Boosting Classifier (Balanced XGBoost) performing 3-class categorization (`Low`, `Medium`, `High`).
2. **Model B (Lighting Condition Classifier):** Random Forest Classifier performing 3-class categorization (`Dark`, `Low`, `Good`).
3. **Model C (Continuous Lux Regressor):** Random Forest Regressor estimating continuous illuminance $L \in [0.1, 9500.0]\,\text{lux}$.

---

### 2. Model A: Balanced XGBoost Crowd Density Classifier

#### 2.1 Feature Engineering and Preprocessing Pipeline
- **Raw Input Features:**
  - `location` (Categorical, 17 campus landmarks)
  - `hour_of_day` (Numerical float, $0.0$ to $23.0$)
  - `day_of_week` (Categorical, 7 weekday names)
  - `weekend_indicator` (Binary integer, $0$ or $1$)
- **Preprocessing Transformer (`ColumnTransformer`):**
  - Categorical variables (`location`, `day_of_week`) are transformed via `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`.
  - Numerical variables (`hour_of_day`, `weekend_indicator`) pass through unscaled.
  - Total transformed feature dimensionality: **26 input dimensions**.
- **Target Encoding:**
  - `Low` $\to 0$
  - `Medium` $\to 1$
  - `High` $\to 2$

#### 2.2 Model Architecture and Hyperparameters
- **Estimator:** `xgboost.XGBClassifier`
- `n_estimators`: $100$
- `max_depth`: $6$
- `learning_rate`: $0.1$
- `objective`: `'multi:softprob'` (Outputs calibrated multi-class probability distributions)
- `eval_metric`: `'mlogloss'`
- `tree_method`: `'auto'`
- `random_state`: $42$

#### 2.3 The Medium-Class Suppression Problem & Cost-Sensitive Solution

##### The Scientific Challenge
During initial technical audits, unweighted classifiers achieved high overall test accuracy ($72.67\%$) but exhibited an unacceptable **$4.48\%$ recall on the `Medium` crowd category** (correctly identifying only 119 out of 2,656 test records).

##### Root Cause Analysis
Because the synthetic dataset encodes daily schedules where either `High` (academic/hostel peaks) or `Low` (late-night curfew) represents the modal baseline, the `Medium` category only occurs as a transitional perturbation ($P \le 0.42$). Under standard unweighted cross-entropy loss minimization, gradient boosting trees naturally learn to suppress the non-modal minority class in favor of the modes.

##### The Mathematically Justified Correction
Rather than fabricating synthetic data or altering the test split, we implemented **cost-sensitive balanced sample weighting** computed exclusively on the training split:

$$w_c = \frac{N_{\text{train}}}{C \cdot N_c}$$

where $N_{\text{train}} = 48,960$, $C = 3$ classes, and $N_c$ is the training sample count for class $c$.

##### Comparative Benchmark Results

| Model Evaluated | Validation Macro F1 | Test Accuracy | Test Macro Precision | Test Macro Recall | Test Macro F1 | Test Medium Recall |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dummy Classifier (Prior)** | $0.2180$ | $48.06\%$ | $0.1602$ | $0.3333$ | $0.2164$ | $0.00\%$ |
| **Logistic Regression (Balanced)** | $0.4184$ | $44.22\%$ | $0.4245$ | $0.4294$ | $0.4125$ | $43.68\%$ |
| **Random Forest (Balanced)** | $0.6468$ | $71.91\%$ | $0.6566$ | $0.6545$ | $0.6499$ | $38.14\%$ |
| **Unweighted XGBoost Baseline** | $0.5782$ | $72.67\%$ | $0.6845$ | $0.5412$ | $0.5702$ | $4.48\%$ |
| **Balanced XGBoost (Selected)** | **$0.6706$** | **$71.32\%$** | **$0.6693$** | **$0.6761$** | **$0.6722$** | **$42.39\%$** |

*Outcome:* Medium class recall achieved a **$9.5\times$ relative improvement** ($4.48\% \to 42.39\%$), lifting Test Macro F1 by **$+10.2$ percentage points** while preserving high overall accuracy ($71.32\%$).

#### 2.4 Test Confusion Matrix & Per-Class Metrics (Model A)

```text
                     Predicted Low    Predicted Medium    Predicted High    Total Support
True Low (0)             3,112               293                 296            3,701
True Medium (1)            715             1,126                 815            2,656
True High (2)              141             1,250               4,492            5,883
Total Predicted          3,968             2,669               5,603           12,240
```

- **Class `Low`:** Precision = $78.43\%$, Recall = $84.09\%$, F1 = $0.8116$
- **Class `Medium`:** Precision = $42.19\%$, Recall = $42.39\%$, F1 = $0.4229$
- **Class `High`:** Precision = $80.17\%$, Recall = $76.36\%$, F1 = $0.7822$

---

### 3. Model B: Random Forest Lighting Condition Classifier

#### 3.1 Model Architecture and Hyperparameters
- **Estimator:** `sklearn.ensemble.RandomForestClassifier`
- `n_estimators`: $100$
- `max_depth`: $10$
- `criterion`: `'gini'`
- `random_state`: $42$
- `n_jobs`: $-1$
- **Target Encoding:** `Dark` $\to 0$, `Low` $\to 1$, `Good` $\to 2$.

#### 3.2 Benchmark Comparison

| Model | Val Accuracy | Val Macro F1 | Test Accuracy | Test Macro Precision | Test Macro Recall | Test Macro F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dummy Classifier** | $71.15\%$ | $0.2771$ | $70.95\%$ | $0.2365$ | $0.3333$ | $0.2767$ |
| **Logistic Regression**| $73.53\%$ | $0.3943$ | $73.42\%$ | $0.8271$ | $0.4037$ | $0.3957$ |
| **Random Forest (Selected)**| **$88.95\%$** | **$0.7950$** | **$88.46\%$** | **$0.8265$** | **$0.7527$** | **$0.7833$** |
| **Gradient Boosting (XGB)** | $88.95\%$ | $0.7950$ | $88.46\%$ | $0.8265$ | $0.7527$ | $0.7833$ |

#### 3.3 Test Confusion Matrix & Per-Class Metrics (Model B)

```text
                     Predicted Dark    Predicted Low    Predicted Good    Total Support
True Dark (0)              915               460               92             1,467
True Low (1)               142             1,356              591             2,089
True Good (2)               23               104            8,557             8,684
Total Predicted          1,080             1,920            9,240            12,240
```

- **Class `Dark`:** Precision = $84.72\%$, Recall = $62.37\%$, F1 = $0.7185$
- **Class `Low`:** Precision = $70.62\%$, Recall = $64.91\%$, F1 = $0.6765$
- **Class `Good`:** Precision = $92.61\%$, Recall = $98.54\%$, F1 = $0.9548$

---

### 4. Model C: Continuous Illuminance (Lux) Regressor

#### 4.1 Architecture and Purpose
Operating alongside discrete lighting categories, Model C estimates continuous lux values $L \in [0.1, 9500.0]\,\text{lx}$ to provide fine-grained environmental context.
- **Estimator:** `sklearn.ensemble.RandomForestRegressor`
- `n_estimators`: $100$
- `max_depth`: $10$
- `random_state`: $42$

#### 4.2 Empirical Regression Metrics

| Metric | Validation Set (Days 121–150) | Chronological Test Set (Days 151–180) |
| :--- | :---: | :---: |
| **Mean Absolute Error (MAE)** | $381.18\,\text{lx}$ | **$378.10\,\text{lx}$** |
| **Root Mean Squared Error (RMSE)**| $653.76\,\text{lx}$ | **$650.54\,\text{lx}$** |
| **Coefficient of Determination ($R^2$)**| $0.7507$ | **$0.7571$** |

*Interpretation:* The regressor explains **$75.71\%$ of variance** in synthetic diurnal illuminance, accurately capturing midday sunlight surges ($> 2,500\,\text{lx}$), indoor lamp levels (~$200\,\text{lx}$), and nighttime darkness ($< 15\,\text{lx}$).

---

### 5. Situational Context Scoring Formulation

The continuous environmental vulnerability score $P_{\text{env}} \in [0.05, 1.00]$ is computed deterministically from model outputs:

$$P_{\text{env}} = \min\left(1.0, \max\left(0.05, \text{Factor}_{\text{time}} + \text{Factor}_{\text{light}} + \text{Factor}_{\text{crowd}}\right)\right)$$

#### Configured Factor Constants
1. **Time Factor ($\text{Factor}_{\text{time}}$):**
   - Night ($22:00 \le \text{hour} < 05:00$): $0.25$
   - Twilight ($18:00 \le \text{hour} < 22:00$ or $05:00 \le \text{hour} < 07:00$): $0.12$
   - Daytime ($07:00 \le \text{hour} < 18:00$): $0.05$
2. **Lighting Factor ($\text{Factor}_{\text{light}}$):**
   - Predicted `Dark`: $0.35$ (High environmental hazard)
   - Predicted `Low`: $0.18$
   - Predicted `Good`: $0.05$
3. **Crowd Factor ($\text{Factor}_{\text{crowd}}$):**
   - Predicted `Low` (High isolation): $0.32$
   - Predicted `Medium`: $0.15$
   - Predicted `High` (Natural collective witness surveillance): $0.02$
