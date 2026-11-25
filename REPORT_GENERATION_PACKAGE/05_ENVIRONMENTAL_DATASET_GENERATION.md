# Multimodal Threat Prediction Engine — Version 1.0
## Technical Documentation: Synthetic Environmental Dataset Generation and Simulation Rules

---

### 1. Dataset Generation Methodology & Academic Transparency
The environmental branch models situational vulnerability by predicting campus crowd density, lighting conditions, and ambient illumination in lux based on diurnal calendar schedules and physical landmark geography.

#### Statement of Academic Integrity and Provenance
> [!IMPORTANT]
> **SYNTHETIC PROVENANCE DISCLOSURE:** The environmental dataset (`BIT_Mesra_Synthetic_Environmental_Dataset.csv`) is a **synthetically generated diurnal schedule simulation**. It is created by algorithmically expanding the baseline BIT Mesra campus schedule rules across 180 simulated days using stochastic Gaussian and multinomial perturbations. **It does NOT represent physical optical lux measurements or camera-based real-time crowd censuses.** The models evaluate expected situational vulnerability based on regular campus operating schedules.

---

### 2. Dataset Schema and Split Specifications

#### 2.1 Dataset Parameters
- **Simulated Duration:** 180 consecutive days (January 1, 2026 to June 29, 2026).
- **Locations Surveyed:** 17 BIT Mesra campus landmarks.
- **Diurnal Granularity:** 24 hourly time slots per day ($00:00$ to $23:00$ IST).
- **Total Observations:** $180 \times 17 \times 24 = \mathbf{73,440\,\text{rows}}$.
- **Random Seed:** `seed = 42` (guaranteeing exact deterministic reproducibility).

#### 2.2 Chronological Partitioning (No Data Leakage)
To evaluate realistic temporal forecasting, the dataset uses a strict **chronological split** rather than random shuffling:
- **Training Set (Days 1 to 120):** $48,960\,\text{records}$ ($66.67\%$).
- **Validation Set (Days 121 to 150):** $12,240\,\text{records}$ ($16.67\%$).
- **Test Set (Days 151 to 180):** $12,240\,\text{records}$ ($16.67\%$).

#### 2.3 Column Schema and Definitions

| Column Name | Data Type | Physical Unit | Permissible Values / Range | Description |
| :--- | :--- | :--- | :--- | :--- |
| `simulated_day` | `int64` | `days` | $1$ to $180$ | Sequential day index |
| `date` | `string` | `YYYY-MM-DD` | `2026-01-01` to `2026-06-29` | Gregorian calendar date |
| `day_of_week` | `string` | `name` | `Monday` to `Sunday` | Day of week |
| `day_of_week_num`| `int64` | `index` | $0$ (Monday) to $6$ (Sunday) | Numeric day index |
| `weekend_indicator`| `int64` | `binary` | $0$ (Weekday), $1$ (Weekend) | Academic schedule flag |
| `hour_of_day` | `int64` | `hours` | $0$ to $23$ | Diurnal hour in IST |
| `location` | `string` | `categorical`| 17 Campus Landmark names | Specific campus zone |
| `simulated_crowd_density_category` | `string` | `target` | `Low`, `Medium`, `High` | Model A target class |
| `simulated_lighting_category` | `string` | `target` | `Dark`, `Low`, `Good` | Model B target class |
| `simulated_lighting_lux_estimate` | `float64` | `lux (lx)` | $0.10$ to $9,500.00\,\text{lx}$ | Model C regression target |
| `split` | `string` | `metadata` | `Train`, `Validation`, `Test` | Evaluation split identifier |
| `provenance` | `string` | `metadata` | `SYNTHETIC_SIMULATION_RULE_GENERATED` | Data integrity watermark |

---

### 3. Environmental Baseline Schedule Rules for All 17 Locations

The simulation derives from the official baseline BIT Mesra campus schedule workbook (`BIT_Mesra_Campus_Crowd_Dataset(1).xlsx`). The table below outlines the deterministic baseline operating rules for each location:

| Landmark Name | Academic Zone Category | Baseline Daytime Crowd (08:00–17:00) | Baseline Evening Crowd (18:00–22:00) | Baseline Night Crowd (23:00–05:00) | Daytime Lighting | Nighttime Lighting |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Main Building** | Administrative Hub | `High` (Lectures & Admin) | `Low` (Offices closed) | `Low` (Deserted) | `Good` | `Good` (CCTV floodlights) |
| **R&D / Library** | Academic Center | `High` (Study & Labs) | `High` (Evening study) | `Low` (Curfew closed) | `Good` | `Good` (Exterior lamps) |
| **Hostel 1 (H1)** | Residential Sector | `Low` (In classes) | `High` (Mess & Common room)| `High` (In rooms) | `Good` | `Low` (Hostel alleyways) |
| **Hostel 2 (H2)** | Residential Sector | `Low` (In classes) | `High` (Mess & Common room)| `High` (In rooms) | `Good` | `Low` (Hostel alleyways) |
| **Hostel 3 (H3)** | Residential Sector | `Low` (In classes) | `High` (Mess & Common room)| `High` (In rooms) | `Good` | `Low` (Hostel alleyways) |
| **Hostel 4 (H4)** | Residential Sector | `Low` (In classes) | `High` (Mess & Common room)| `High` (In rooms) | `Good` | `Low` (Hostel alleyways) |
| **Girls Hostel / H8-H9**| Residential Sector | `Low` (In classes) | `High` (Hostel perimeter) | `High` (In rooms) | `Good` | `Good` (Enhanced lighting) |
| **PMC** | Industrial Outpost | `Low` (Remote workshops) | `Low` (Deserted path) | `Low` (Unlit corridor) | `Good` | `Good` (Outpost gate) |
| **Hostel 6 (H6)** | Residential Sector | `Low` (In classes) | `High` (Mess hours) | `High` (In rooms) | `Good` | `Low` (Hostel alleyways) |
| **Hostel 7 (H7)** | Residential Sector | `Low` (In classes) | `High` (Mess hours) | `High` (In rooms) | `Good` | `Low` (Hostel alleyways) |
| **Dispensary** | Medical Center | `High` (OPD Hours) | `Low` (Emergency only) | `Low` (Emergency only) | `Good` | `Good` (Medical porch) |
| **Hostel 10 (H10)**| Isolated Boundary | `Low` (In classes) | `High` (Mess hours) | `Low` (Perimeter unlit) | `Good` | `Dark` (Forest boundary) |
| **Hostel 11 (H11)**| Isolated Boundary | `Low` (In classes) | `High` (Mess hours) | `Low` (Perimeter unlit) | `Good` | `Dark` (Forest boundary) |
| **Hostel 12 (H12)**| Isolated Boundary | `Low` (In classes) | `High` (Mess hours) | `Low` (Perimeter unlit) | `Good` | `Dark` (Forest boundary) |
| **Hostel 13 (H13)**| Isolated Boundary | `Low` (In classes) | `High` (Mess hours) | `Low` (Perimeter unlit) | `Good` | `Dark` (Forest boundary) |
| **Sports Complex**| Open Recreational | `Low` (Class hours) | `High` (Sports & Gym) | `Low` (Closed grounds) | `Good` | `Low` (Perimeter track) |
| **Main Gate** | Security Checkpoint | `High` (Traffic ingress) | `High` (Return curfews) | `Low` (Guarded ingress) | `Good` | `Good` (24/7 Security) |

---

### 4. Mathematical Simulation Logic and Perturbation Equations

#### 4.1 Crowd Density Stochastic Simulation
Let $\text{Base}_{\text{crowd}} \in \{\text{Low}, \text{High}\}$ be the deterministic schedule baseline. The simulated category is drawn from a multinomial distribution:

$$\text{Crowd}_{\text{sim}} \sim \text{Multinomial}\left(1, [p_{\text{Low}}, p_{\text{Med}}, p_{\text{High}}]\right)$$

The probability vector $\mathbf{p} = [p_{\text{Low}}, p_{\text{Med}}, p_{\text{High}}]$ adapts to schedule context:
1. **If Baseline is `High`:**
   - On weekends in academic zones (Main Building, R&D):
     $$\mathbf{p} = [0.60, 0.30, 0.10]$$
   - During transition hours ($07:00, 08:00, 13:00, 14:00, 20:00, 21:00$):
     $$\mathbf{p} = [0.08, 0.42, 0.50]$$
   - Standard peak schedule:
     $$\mathbf{p} = [0.05, 0.15, 0.80]$$
2. **If Baseline is `Low`:**
   - On weekends at Sports Complex during activity hours ($07:00–10:00, 16:00–19:00$):
     $$\mathbf{p} = [0.15, 0.45, 0.40]$$
   - During shoulder hours ($06:00, 12:00, 22:00$):
     $$\mathbf{p} = [0.65, 0.30, 0.05]$$
   - Standard off-peak:
     $$\mathbf{p} = [0.88, 0.10, 0.02]$$

#### 4.2 Lighting Condition Stochastic Simulation
Natural sunlight and artificial lighting infrastructure govern lighting categories:
- **Daytime ($06:00 \le \text{hour} < 18:00$):**
  $$\mathbf{p}_{\text{light}} = [0.005, 0.035, 0.960] \quad [\text{Dark}, \text{Low}, \text{Good}]$$
- **Nighttime ($18:00 \le \text{hour} < 06:00$):**
  - If baseline is `Good` (Main Gate, Dispensary, PMC):
    $$\mathbf{p}_{\text{light}} = [0.02, 0.12, 0.86]$$
  - If baseline is `Low` (Hostels H1–H7, Sports Complex):
    $$\mathbf{p}_{\text{light}} = [0.22, 0.72, 0.06]$$
  - If baseline is `Dark` (Perimeter Hostels H10–H13):
    $$\mathbf{p}_{\text{light}} = [0.85, 0.13, 0.02]$$

#### 4.3 Continuous Illuminance (Lux) Generation Equations
Continuous illuminance $L$ (in lux) is generated from clipped Gaussian distributions matching empirical photometric standards:

$$L = \begin{cases}
\text{clip}\left(\mathcal{N}(4.0, 2.5^2), 0.1, 15.0\right) & \text{if } \text{Lighting} = \text{Dark} \\
\text{clip}\left(\mathcal{N}(55.0, 18.0^2), 15.1, 120.0\right) & \text{if } \text{Lighting} = \text{Low} \\
\text{clip}\left(\mathcal{N}(2500.0, 800.0^2), 300.0, 9500.0\right) & \text{if } \text{Lighting} = \text{Good} \land \text{Daytime} \\
\text{clip}\left(\mathcal{N}(220.0, 45.0^2), 121.0, 450.0\right) & \text{if } \text{Lighting} = \text{Good} \land \text{Nighttime}
\end{cases}$$

---

### 5. Empirical Dataset Summary Statistics

#### 5.1 Target Class Distributions ($N = 73,440$)

```text
Crowd Density Categories:
• High:   35,394 observations (48.19%)
• Low:    22,126 observations (30.13%)
• Medium: 15,920 observations (21.68%)

Lighting Condition Categories:
• Good:   52,143 observations (71.00%)
• Low:    12,476 observations (16.99%)
• Dark:    8,821 observations (12.01%)
```

#### 5.2 Numerical Feature Summary Table

| Metric | Simulated Day | Hour of Day | Weekend Indicator | Simulated Lux Estimate (lx) |
| :--- | :---: | :---: | :---: | :---: |
| **Mean** | $90.50$ | $11.50$ | $0.2857$ | $1,288.75$ |
| **Std Dev** | $51.96$ | $6.92$ | $0.4518$ | $1,324.62$ |
| **Minimum** | $1.00$ | $0.00$ | $0.0000$ | $0.10$ |
| **25% Quantile** | $45.75$ | $5.75$ | $0.0000$ | $6.82$ |
| **Median (50%)** | $90.50$ | $11.50$ | $0.0000$ | $245.10$ |
| **75% Quantile** | $135.25$ | $17.25$ | $1.0000$ | $2,480.12$ |
| **Maximum** | $180.00$ | $23.00$ | $1.0000$ | $9,450.00$ |
