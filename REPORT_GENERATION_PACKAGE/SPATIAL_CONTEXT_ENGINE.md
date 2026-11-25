# Multimodal Threat Prediction Engine — Version 1.0
## Technical Documentation: Spatial Context Engine, BIT Mesra GIS, and Geofencing Logic

---

### 1. Spatial Branch Architecture and Philosophy
The Spatial Context branch evaluates real-time geographic vulnerability for an individual on the campus of the Birla Institute of Technology (BIT), Mesra. 

Unlike purely statistical machine learning classifiers that predict subjective threat categories, the core Spatial Engine operates as a **deterministic geospatial inference engine**. It computes exact physical distances, verifies geofencing boundaries, detects GPS ambiguity, and evaluates a transparent situational vulnerability score based on verified campus infrastructure parameters (proximity to guarded security posts, regional isolation indices, and time-adjusted illumination).

---

### 2. BIT Mesra Campus Landmarks Database (All 17 Surveyed Locations)

The campus layout was surveyed using OpenStreetMap (OSM) and campus administrative records. The table below details all 17 supported locations, their verified coordinates, detection radii, physical categories, base risk parameters, and distances to security infrastructure:

| Index | Landmark ID | Official Landmark Name | Latitude ($^\circ\text{N}$) | Longitude ($^\circ\text{E}$) | Detection Radius ($r$) | Location Category | Base Risk ($R_0$) | Base Isolation ($I_0$) | Distance to Security Post |
| :---: | :--- | :--- | :---: | :---: | :---: | :--- | :---: | :---: | :---: |
| **1** | `main_building` | **Main Building (Clock Tower)** | $23.412300$ | $85.439900$ | $150\,\text{m}$ | `administrative_hub` | $0.08$ | $0.05$ | $20\,\text{m}$ |
| **2** | `rnd` | **R&D / Central Library** | $23.413500$ | $85.438500$ | $150\,\text{m}$ | `academic_zone` | $0.12$ | $0.15$ | $50\,\text{m}$ |
| **3** | `h1` | **Hostel 1 (H1)** | $23.414209$ | $85.440313$ | $120\,\text{m}$ | `residential_hostel` | $0.25$ | $0.25$ | $120\,\text{m}$ |
| **4** | `h2` | **Hostel 2 (H2)** | $23.413707$ | $85.438484$ | $120\,\text{m}$ | `residential_hostel` | $0.25$ | $0.25$ | $130\,\text{m}$ |
| **5** | `h3` | **Hostel 3 (H3)** | $23.416120$ | $85.439653$ | $120\,\text{m}$ | `residential_hostel` | $0.28$ | $0.30$ | $150\,\text{m}$ |
| **6** | `h4` | **Hostel 4 (H4)** | $23.415602$ | $85.437798$ | $120\,\text{m}$ | `residential_hostel` | $0.28$ | $0.30$ | $170\,\text{m}$ |
| **7** | `girls_hostel` | **Girls Hostel Area / H8-H9** | $23.414000$ | $85.441500$ | $150\,\text{m}$ | `residential_hostel` | $0.20$ | $0.20$ | $80\,\text{m}$ |
| **8** | `pmc` | **PMC (Polytechnic & Medical)** | $23.416205$ | $85.440121$ | $150\,\text{m}$ | `unlit_industrial_alley` | $0.65$ | $0.80$ | $480\,\text{m}$ |
| **9** | `h6` | **Hostel 6 (H6)** | $23.422811$ | $85.433202$ | $120\,\text{m}$ | `residential_hostel` | $0.30$ | $0.35$ | $200\,\text{m}$ |
| **10** | `h7` | **Hostel 7 (H7)** | $23.424604$ | $85.432130$ | $120\,\text{m}$ | `residential_hostel` | $0.32$ | $0.40$ | $230\,\text{m}$ |
| **11** | `dispensary` | **Dispensary / Health Centre** | $23.410500$ | $85.438800$ | $100\,\text{m}$ | `medical_center` | $0.15$ | $0.18$ | $100\,\text{m}$ |
| **12** | `h10` | **Hostel 10 (H10)** | $23.419115$ | $85.434912$ | $120\,\text{m}$ | `isolated_hostel_boundary`| $0.45$ | $0.70$ | $400\,\text{m}$ |
| **13** | `h11` | **Hostel 11 (H11)** | $23.418032$ | $85.435418$ | $120\,\text{m}$ | `isolated_forest_boundary`| $0.50$ | $0.78$ | $450\,\text{m}$ |
| **14** | `h12` | **Hostel 12 (H12)** | $23.418919$ | $85.434311$ | $120\,\text{m}$ | `isolated_forest_boundary`| $0.55$ | $0.85$ | $520\,\text{m}$ |
| **15** | `h13` | **Hostel 13 (H13)** | $23.417810$ | $85.434655$ | $120\,\text{m}$ | `isolated_forest_boundary`| $0.60$ | $0.92$ | $600\,\text{m}$ |
| **16** | `sports_complex`| **Sports Complex & Gym Ground** | $23.425020$ | $85.434623$ | $200\,\text{m}$ | `open_recreational` | $0.35$ | $0.55$ | $250\,\text{m}$ |
| **17** | `main_gate` | **Main Gate (Security Post 1)** | $23.405500$ | $85.435000$ | $150\,\text{m}$ | `security_checkpoint` | $0.05$ | $0.02$ | $0\,\text{m}$ |

*Campus Center Benchmark:* $23.412300^\circ\,\text{N}, 85.439900^\circ\,\text{E}$ (Main Building Clock Tower).

---

### 3. Mathematical Formulation of Geospatial Algorithms

#### 3.1 Haversine Great-Circle Distance
Given a user position $(\phi_1, \lambda_1)$ and a landmark position $(\phi_2, \lambda_2)$ in radians, the great-circle surface distance $d$ in meters is computed as:

$$\Delta \phi = \phi_2 - \phi_1, \quad \Delta \lambda = \lambda_2 - \lambda_1$$

$$a = \sin^2\left(\frac{\Delta \phi}{2}\right) + \cos(\phi_1)\cos(\phi_2)\sin^2\left(\frac{\Delta \lambda}{2}\right)$$

$$c = 2 \cdot \text{atan2}\left(\sqrt{a}, \sqrt{1 - a}\right)$$

$$d = R \cdot c$$

where the mean Earth radius constant is set to $R = 6,371,000.0\,\text{meters}$.

#### 3.2 Maximum Geofence Enforcement ($350.0\,\text{m}$)
The system enforces an explicit campus safety perimeter:
- Let $d_1 = \min_{i \in \{1,\dots,17\}} d(\mathbf{x}_{\text{user}}, \mathbf{x}_{\text{landmark}, i})$.
- If $d_1 > 350.0\,\text{meters}$, the user coordinate is flagged as `OUTSIDE_CAMPUS_GEOFENCE`.
- The engine halts automatic landmark association, emits an informational warning banner, and requires the operator or student to select a campus landmark manually from the supported catalog.

#### 3.3 Defensive GPS Ambiguity Resolution
In physical deployments, consumer smartphones frequently report location uncertainty (e.g., GPS accuracy error radii between $\pm 15\,\text{m}$ and $\pm 50\,\text{m}$). When an individual is located in adjacent hostel clusters (e.g., between H10 and H12), arbitrary nearest-neighbor selection can lead to incorrect contextual analysis.

To eliminate this:
1. Sort all 17 landmarks by distance: $d_1 \le d_2 \le \dots \le d_{17}$.
2. Compute the inter-landmark distance differential: $\Delta d = d_2 - d_1$.
3. Determine the dynamic ambiguity threshold:
   $$\tau_{\text{ambiguity}} = \max\left(25.0\,\text{m}, \text{accuracy}_{\text{reported}}\right)$$
4. **Decision Rule:**
   $$\text{If } \Delta d \le \tau_{\text{ambiguity}} \implies \text{Status: } \text{AMBIGUOUS\_LOCATION}$$
5. The system flags candidates $LM_1$ and $LM_2$ in the UI and requires explicit manual landmark confirmation before proceeding with downstream threat fusion.

#### 3.4 Situational Spatial Vulnerability Formulation
When a valid landmark is resolved, the continuous spatial context score $P_{\text{spatial}} \in [0.0, 1.0]$ is evaluated deterministically:

1. **Center Distance & Spatial Isolation ($I_{\text{spatial}}$):**
   $$d_{\text{center}} = \text{haversine}(\phi_{\text{user}}, \lambda_{\text{user}}, \phi_{\text{center}}, \lambda_{\text{center}})$$
   $$I_{\text{spatial}} = \min\left(1.0, \max\left(0.0, I_0 + \frac{d_{\text{center}}}{1000.0} \times 0.2\right)\right)$$
2. **Diurnal Time Penalty ($P_{\text{night}}$):**
   $$P_{\text{night}} = \begin{cases} 0.25 & \text{if } \text{hour} \ge 19.0 \text{ or } \text{hour} \le 5.0 \\ 0.00 & \text{otherwise} \end{cases}$$
3. **Composite Spatial Vulnerability ($P_{\text{spatial}}$):**
   $$P_{\text{spatial}} = \min\left(1.0, \max\left(0.0, R_0 + P_{\text{night}} \times I_{\text{spatial}}\right)\right)$$

---

### 4. Browser Geolocation API Integration and Data Contract
The Streamlit application interfaces with the client device's physical GPS hardware via a custom HTML5 Geolocation iframe component (`app/components/gps/index.html`).

#### Geolocation Workflow
1. The frontend executes `navigator.geolocation.watchPosition()` with `enableHighAccuracy: true`, `timeout: 10000`, and `maximumAge: 5000`.
2. Coordinates are packaged into a JSON payload:
   ```json
   {
     "latitude": 23.418919,
     "longitude": 85.434311,
     "accuracy": 18.5,
     "timestamp": 1795528800000
   }
   ```
3. Delivered across the iframe barrier to Streamlit session state `st.session_state["location_res"]`.

#### Standardized Location Status Codes

| Status Code | Trigger Condition | System Behavior |
| :--- | :--- | :--- |
| `RESOLVED_GPS` | $d_1 \le 350.0\,\text{m}$ and $\Delta d > \tau_{\text{ambiguity}}$ | Nearest landmark accepted; coordinates plotted on Folium map. |
| `AMBIGUOUS_LOCATION` | $\Delta d \le \max(25\,\text{m}, \text{accuracy})$ | Ambiguity alert displayed; manual selection dropdown activated. |
| `OUTSIDE_CAMPUS_GEOFENCE` | $d_1 > 350.0\,\text{m}$ | Boundary alert displayed; automatic fallback to manual catalog. |
| `GPS_UNAVAILABLE` | Coordinates `None`, empty, or permission denied | Informs user; defaults gracefully to manual landmark selection. |
| `LOW_ACCURACY` | Reported accuracy $> 100.0\,\text{m}$ | Displays caution pill; instructs user to wait for satellite lock. |
| `CONNECTED` | Normal GPS streaming active | Green telemetry status badge displayed. |

---

### 5. Worked Numerical Examples Using Implemented Rules

#### Example 1: Daytime Student at Main Building (Low Spatial Vulnerability)
- **Inputs:** Lat = $23.412350$, Lon = $85.439920$, Accuracy = $12.0\,\text{m}$, Hour = $14:00$ (2:00 PM).
- **Distance Calculation:**
  - $d(\text{Main Building}) = 5.9\,\text{m}$.
  - $d(\text{R\&D}) = 193.2\,\text{m}$.
  - Nearest landmark: Main Building ($d_1 = 5.9\,\text{m} \le 150\,\text{m}$ radius).
- **Geofence Check:** $5.9\,\text{m} \le 350\,\text{m} \implies \text{Within Geofence}$.
- **Ambiguity Check:** $\Delta d = 193.2 - 5.9 = 187.3\,\text{m} > \max(25, 12) = 25.0\,\text{m} \implies \text{Unambiguous}$.
- **Status:** `RESOLVED_GPS`.
- **Score Evaluation:**
  - $d_{\text{center}} = 5.9\,\text{m}$.
  - $I_{\text{spatial}} = 0.05 + (5.9 / 1000) \times 0.2 = 0.0512$.
  - Hour = $14.0 \implies P_{\text{night}} = 0.00$.
  - $P_{\text{spatial}} = R_0 + 0 = \mathbf{0.0800}$ ($8.0\%$ spatial vulnerability).

#### Example 2: Late-Night Student at Hostel 13 Perimeter (High Spatial Vulnerability)
- **Inputs:** Lat = $23.417810$, Lon = $85.434655$, Accuracy = $15.0\,\text{m}$, Hour = $23:30$ (11:30 PM).
- **Distance Calculation:**
  - Nearest landmark: Hostel H13 ($d_1 = 0.0\,\text{m}$).
  - Second nearest: Hostel H12 ($d_2 = 127.8\,\text{m}$).
- **Geofence & Ambiguity:** $\Delta d = 127.8\,\text{m} > 25.0\,\text{m} \implies \text{Unambiguous}$.
- **Status:** `RESOLVED_GPS`.
- **Score Evaluation:**
  - $d_{\text{center}} \approx 814.5\,\text{m}$ from Main Building.
  - $I_{\text{spatial}} = 0.92 + (814.5 / 1000) \times 0.2 = 0.92 + 0.1629 = 1.00$ (clipped to $1.0$).
  - Hour = $23.5 \implies P_{\text{night}} = 0.25$.
  - $P_{\text{spatial}} = 0.60 + 0.25 \times 1.00 = \mathbf{0.8500}$ ($85.0\%$ spatial vulnerability).

#### Example 3: Equidistant Ambiguity Between H10 and H12
- **Inputs:** Lat = $23.419017$, Lon = $85.434612$, Accuracy = $30.0\,\text{m}$, Hour = $20:00$.
- **Distance Calculation:**
  - $d(\text{H10}) = 32.4\,\text{m}$.
  - $d(\text{H12}) = 32.8\,\text{m}$.
  - $\Delta d = |32.4 - 32.8| = 0.4\,\text{m}$.
- **Ambiguity Test:** $\tau_{\text{ambiguity}} = \max(25.0, 30.0) = 30.0\,\text{m}$.
  - $\Delta d = 0.4\,\text{m} \le 30.0\,\text{m} \implies \mathbf{AMBIGUOUS\_LOCATION}$.
- **Action:** Halts automated geofencing; UI prompts user to confirm whether they are at H10 or H12.

#### Example 4: User Located Off-Campus (Ranchi Railway Station)
- **Inputs:** Lat = $23.350000$, Lon = $85.320000$ (~$15\,\text{km}$ from BIT Mesra).
- **Distance Calculation:**
  - Nearest landmark: Main Gate ($d_1 \approx 13,420\,\text{m}$).
- **Geofence Test:** $13,420\,\text{m} > 350.0\,\text{m} \implies \mathbf{OUTSIDE\_CAMPUS\_GEOFENCE}$.
- **Action:** System warns that GPS exceeds supported campus boundary and engages manual landmark selection mode.
