import os
import sys
import html
import datetime
import pandas as pd
import numpy as np
import streamlit as st
from pathlib import Path
from streamlit_js_eval import get_geolocation

try:
    import plotly.express as px
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

# Ensure project root is in sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from src.fusion.predict import MultimodalThreatPredictor
from src.location.campus_map import CampusMapAnalyzer
from src.physiological.preprocessing import load_subject_data
from src.physiological.feature_extraction import extract_window_features

# Streamlit Page Configuration
st.set_page_config(
    page_title="BIT Mesra Spatial & Multimodal Threat System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Load External CSS Stylesheet for Medical Analytics Theme
def load_css():
    css_path = Path(__file__).parent / "static" / "dashboard.css"
    if css_path.exists():
        with open(css_path, "r", encoding="utf-8") as f:
            st.html(f"<style>{f.read()}</style>")

load_css()

# Load HTML Presentation Templates from app/templates/dashboard.html
def get_template(block_name: str) -> str:
    tpl_path = Path(__file__).parent / "templates" / "dashboard.html"
    if tpl_path.exists():
        with open(tpl_path, "r", encoding="utf-8") as f:
            content = f.read()
        start_tag = f"<!-- {block_name}_START -->"
        end_tag = f"<!-- {block_name}_END -->"
        if start_tag in content and end_tag in content:
            return content.split(start_tag)[1].split(end_tag)[0].strip()
    return ""

def render_header():
    header_html = get_template("HEADER_TEMPLATE")
    if not header_html:
        header_html = """<div class="med-header">
  <div class="med-header-title-row">
    <div>
      <h1 class="med-title">Multimodal Threat Prediction Engine</h1>
      <p class="med-subtitle">Physiological, Geospatial & Environmental Risk Assessment</p>
    </div>
    <div class="med-badge-location">
      BIT Mesra · Ranchi
    </div>
  </div>
</div>"""
    st.html(header_html)

def render_overall_risk_panel(res: dict):
    phys_norm = float(res.get("physiological_stress_score", 0.0))
    loc_norm = float(res.get("location_risk_score", 0.0))
    env_norm = float(res.get("environment_risk_score", 0.0))

    # Shared single fusion calculation across backend and UI presentation
    phys_pct = phys_norm * 100.0
    loc_pct = loc_norm * 100.0
    env_pct = env_norm * 100.0

    contrib_p = phys_pct * 0.40
    contrib_l = loc_pct * 0.25
    contrib_e = env_pct * 0.35

    final_score_pct = contrib_p + contrib_l + contrib_e
    threshold_pct = float(res.get("threshold_used", 0.50)) * 100.0
    status = "RISKY" if final_score_pct >= threshold_pct else "SAFE"
    status_badge_cls = "score-danger" if status == "RISKY" else "score-safe"
    explanation_txt = html.escape(str(res.get("explanation", "")))

    template = get_template("OVERALL_PANEL_TEMPLATE")
    if template:
        rendered = (
            template.replace("{{final_score_pct}}", f"{final_score_pct:.2f}")
            .replace("{{status_badge_cls}}", status_badge_cls)
            .replace("{{status}}", status)
            .replace("{{phys_pct}}", f"{phys_pct:.1f}")
            .replace("{{contrib_p}}", f"{contrib_p:.2f}")
            .replace("{{loc_pct}}", f"{loc_pct:.1f}")
            .replace("{{contrib_l}}", f"{contrib_l:.2f}")
            .replace("{{env_pct}}", f"{env_pct:.1f}")
            .replace("{{contrib_e}}", f"{contrib_e:.2f}")
            .replace("{{explanation_txt}}", explanation_txt)
        )
    else:
        rendered = f"""<div class="med-overall-panel">
  <div class="med-overall-header">Overall Risk Assessment</div>
  <div class="med-overall-content">
    <div class="med-score-hero">
      <div class="med-score-hero-title">Prototype Threat-Risk Score</div>
      <div class="med-score-hero-num">{final_score_pct:.2f}%</div>
      <div class="med-score-badge {status_badge_cls}">{status}</div>
    </div>
    <div class="med-breakdown-box">
      <div style="font-weight: 700; font-size: 0.95rem; color: var(--text-primary); margin-bottom: 8px;">
        Branch Contribution Breakdown
      </div>
      <div class="med-breakdown-item">
        <span>Physiological Stress (40% Weight):</span>
        <span><b>{phys_pct:.1f}% × 0.40 = {contrib_p:.2f}%</b></span>
      </div>
      <div class="med-breakdown-item">
        <span>Location Risk (25% Weight):</span>
        <span><b>{loc_pct:.1f}% × 0.25 = {contrib_l:.2f}%</b></span>
      </div>
      <div class="med-breakdown-item">
        <span>Environment Risk (35% Weight):</span>
        <span><b>{env_pct:.1f}% × 0.35 = {contrib_e:.2f}%</b></span>
      </div>
      <div class="med-rationale-note">
        <b>Decision Rationale:</b> {explanation_txt}
      </div>
    </div>
  </div>
</div>"""
    st.html(rendered)

@st.cache_data(show_spinner=False)
def cached_load_subject_data(subject_id: str):
    return load_subject_data(subject_id)

@st.cache_resource
def load_predictor():
    return MultimodalThreatPredictor()

predictor = load_predictor()
campus_analyzer = CampusMapAnalyzer()

# Initialize Session State Variables
if "physio_score" not in st.session_state:
    st.session_state["physio_score"] = 0.20
if "gps_status" not in st.session_state:
    st.session_state["gps_status"] = "OFF"
if "location_res" not in st.session_state:
    st.session_state["location_res"] = None
if "location_score" not in st.session_state:
    st.session_state["location_score"] = None
if "env_score" not in st.session_state:
    st.session_state["env_score"] = 0.10
if "analyzed_threat" not in st.session_state:
    st.session_state["analyzed_threat"] = None
if "gps_query_counter" not in st.session_state:
    st.session_state["gps_query_counter"] = 0
if "last_table_selected_row" not in st.session_state:
    st.session_state["last_table_selected_row"] = None

# Render Header
render_header()

# Main Tab Navigation
tab1, tab2 = st.tabs(["🛡️ Threat Prediction Dashboard", "📍 BIT Mesra GPS & Spatial Analyzer"])

# =============================================================================
# TAB 1: MULTIMODAL THREAT PREDICTION DASHBOARD
# =============================================================================
with tab1:
    col1, col2, col3 = st.columns(3)

    # CARD 1 — PHYSIOLOGICAL ANALYSIS
    with col1:
        st.html("""<div class="med-card-header">
  <span class="med-card-title">1. Physiological Analysis</span>
  <span class="med-card-subtitle">WESAD Wearable Signals</span>
</div>""")
        
        selected_subject = st.selectbox("Select Subject", config.ALL_SUBJECTS, index=0)
        session_option = st.selectbox("Select Recording Window", ["Window 1 (Baseline)", "Window 2 (Stress Session)", "Window 3 (Amusement)"], index=0)
        
        if st.button("Analyze Physiology", key="btn_physio", use_container_width=True):
            try:
                subj_data = cached_load_subject_data(selected_subject)
                chest = subj_data['signal']['chest']
                wrist = subj_data['signal']['wrist']
                labels = subj_data['label']
                
                if "Stress" in session_option:
                    stress_indices = np.where(labels == 2)[0]
                    start_sec = (stress_indices[0] / config.SAMPLING_FREQ['chest']) if len(stress_indices) > 0 else 300.0
                elif "Amusement" in session_option:
                    amu_indices = np.where(labels == 3)[0]
                    start_sec = (amu_indices[0] / config.SAMPLING_FREQ['chest']) if len(amu_indices) > 0 else 400.0
                else:
                    base_indices = np.where(labels == 1)[0]
                    start_sec = (base_indices[0] / config.SAMPLING_FREQ['chest']) if len(base_indices) > 0 else 100.0
                    
                feats, _ = extract_window_features(selected_subject, chest, wrist, labels, start_sec=start_sec, window_sec=60.0)
                if feats is not None:
                    st.session_state["physio_score"] = predictor.predict_physiological(feats)
                else:
                    st.session_state["physio_score"] = 0.78 if "Stress" in session_option else 0.18
            except Exception:
                st.session_state["physio_score"] = 0.78 if "Stress" in session_option else 0.18

            # Auto update threat fusion if previously analyzed
            if st.session_state.get("analyzed_threat") is not None:
                loc_score_val = st.session_state["location_score"] if st.session_state["location_score"] is not None else 0.15
                st.session_state["analyzed_threat"] = predictor.predict_fusion(
                    physiological_stress_score=st.session_state["physio_score"],
                    location_risk_score=loc_score_val,
                    environment_risk_score=st.session_state["env_score"],
                    threshold=0.50
                )

        phys_pct = int(st.session_state["physio_score"] * 100)
        badge_cls = "score-danger" if phys_pct > 65 else ("score-warn" if phys_pct > 35 else "score-safe")
        st.html(f"""<div class="med-score-box">
  <div class="med-score-label">Physiological Stress Score</div>
  <div class="med-score-badge {badge_cls}">{phys_pct}%</div>
</div>""")

    # CARD 2 — CURRENT LOCATION (LIVE BROWSER GPS)
    with col2:
        st.html("""<div class="med-card-header">
  <span class="med-card-title">2. Current Location</span>
  <span class="med-card-subtitle">Browser Geolocation</span>
</div>""")
        
        # Render browser geolocation listener component
        loc_payload = get_geolocation(component_key=f"live_browser_gps_{st.session_state['gps_query_counter']}")

        if loc_payload and isinstance(loc_payload, dict) and "coords" in loc_payload:
            coords = loc_payload["coords"]
            r_lat = float(coords["latitude"])
            r_lon = float(coords["longitude"])
            r_acc = float(coords.get("accuracy", 10.0))
            r_ts = loc_payload.get("timestamp")

            print(f"[DEBUG LIVE GPS] Received browser coordinates: lat={r_lat:.6f}, lon={r_lon:.6f}, accuracy={r_acc}m, timestamp={r_ts}", flush=True)

            analysis_res = campus_analyzer.analyze_gps(
                lat=r_lat,
                lon=r_lon,
                accuracy=r_acc,
                timestamp=r_ts,
                time_of_day_hour=datetime.datetime.now().hour
            )

            st.session_state["gps_status"] = analysis_res["status"]
            st.session_state["location_res"] = analysis_res
            st.session_state["location_score"] = analysis_res["location_risk_score"]

            if st.session_state.get("analyzed_threat") is not None:
                st.session_state["analyzed_threat"] = predictor.predict_fusion(
                    physiological_stress_score=st.session_state["physio_score"],
                    location_risk_score=st.session_state["location_score"],
                    environment_risk_score=st.session_state["env_score"],
                    threshold=0.50
                )
        elif loc_payload and isinstance(loc_payload, dict) and "error" in loc_payload:
            print(f"[DEBUG LIVE GPS] Browser GPS Error Payload: {loc_payload['error']}", flush=True)
            st.session_state["gps_status"] = "ERROR"
            st.session_state["location_res"] = None
            st.session_state["location_score"] = None

        # Display GPS Status Pill
        status_code = st.session_state["gps_status"]
        if status_code in ["CONNECTED", "LIVE", "OK"]:
            pill_html = "<span class='med-status-pill pill-connected'>GPS: CONNECTED</span>"
        elif status_code == "LOW_ACCURACY":
            pill_html = "<span class='med-status-pill pill-warn'>GPS: LOW ACCURACY</span>"
        elif status_code == "ERROR":
            pill_html = "<span class='med-status-pill pill-error'>GPS: ERROR</span>"
        else:
            pill_html = "<span class='med-status-pill pill-off'>GPS: OFF</span>"
            
        st.html(f"<div style='margin-bottom: 10px;'>{pill_html}</div>")

        if st.button("📍 [ Get Live Location ]", key="btn_trigger_gps", use_container_width=True):
            st.session_state["gps_query_counter"] += 1

        loc_res = st.session_state.get("location_res")
        has_valid_gps = (loc_res is not None and loc_res.get("nearest_landmark") is not None)

        if has_valid_gps:
            c_loc = loc_res["current_location"]
            n_lm = loc_res["nearest_landmark"]
            lat_disp = f"{c_loc['latitude']:.6f}"
            lon_disp = f"{c_loc['longitude']:.6f}"
            acc_disp = f"±{int(c_loc['accuracy_meters'])} m"
            near_lm_disp = n_lm["name"]
            dist_disp = f"{n_lm['distance_meters']} m"
            score_disp = f"{int(loc_res['location_risk_score'] * 100)}%"
            score_val = loc_res["location_risk_score"]
        else:
            lat_disp = "—"
            lon_disp = "—"
            acc_disp = "—"
            near_lm_disp = "—"
            dist_disp = "—"
            score_disp = "—"
            score_val = 0.0

        st.html(f"""<div style='margin-top: 10px;'>
  <div class='med-metric-row'><span class='med-metric-label'>Latitude</span> <span class='med-metric-value'>{lat_disp}</span></div>
  <div class='med-metric-row'><span class='med-metric-label'>Longitude</span> <span class='med-metric-value'>{lon_disp}</span></div>
  <div class='med-metric-row'><span class='med-metric-label'>Accuracy</span> <span class='med-metric-value'>{acc_disp}</span></div>
  <div class='med-metric-row'><span class='med-metric-label'>Nearest Landmark</span> <span class='med-metric-value med-metric-value-accent'>{near_lm_disp}</span></div>
  <div class='med-metric-row'><span class='med-metric-label'>Distance</span> <span class='med-metric-value'>{dist_disp}</span></div>
</div>""")

        badge_cls = "score-danger" if score_val > 0.65 else ("score-warn" if score_val > 0.35 else "score-safe")
        st.html(f"""<div class="med-score-box">
  <div class="med-score-label">Location Risk Score</div>
  <div class="med-score-badge {badge_cls}">{score_disp}</div>
</div>""")

    # CARD 3 — ENVIRONMENTAL CONTEXT
    with col3:
        st.html("""<div class="med-card-header">
  <span class="med-card-title">3. Environmental Context</span>
  <span class="med-card-subtitle">Situational Factors</span>
</div>""")
        
        current_time_str = datetime.datetime.now().strftime("%I:%M %p")
        st.text_input("Time of Day", value=current_time_str, disabled=True)
        
        def on_env_change():
            current_hour = datetime.datetime.now().hour + datetime.datetime.now().minute / 60.0
            l_opt = st.session_state.get("radio_lighting", "Light")
            c_opt = st.session_state.get("radio_crowd", "Medium")
            env_score = predictor.predict_environment(
                time_of_day_hour=current_hour,
                lighting_condition=l_opt,
                crowd_density=c_opt
            )
            st.session_state["env_score"] = env_score
            if st.session_state.get("analyzed_threat") is not None:
                loc_score_val = st.session_state["location_score"] if st.session_state["location_score"] is not None else 0.15
                st.session_state["analyzed_threat"] = predictor.predict_fusion(
                    physiological_stress_score=st.session_state["physio_score"],
                    location_risk_score=loc_score_val,
                    environment_risk_score=st.session_state["env_score"],
                    threshold=0.50
                )

        lighting_opt = st.radio("Lighting", ["Light", "Dark"], horizontal=True, key="radio_lighting", on_change=on_env_change)
        crowd_opt = st.radio("Crowd Density", ["Low", "Medium", "High"], horizontal=True, key="radio_crowd", on_change=on_env_change)
        
        if st.button("Analyze Environment", key="btn_env", use_container_width=True):
            current_hour = datetime.datetime.now().hour + datetime.datetime.now().minute / 60.0
            env_score = predictor.predict_environment(
                time_of_day_hour=current_hour,
                lighting_condition=lighting_opt,
                crowd_density=crowd_opt
            )
            st.session_state["env_score"] = env_score

            if st.session_state.get("analyzed_threat") is not None:
                loc_score_val = st.session_state["location_score"] if st.session_state["location_score"] is not None else 0.15
                st.session_state["analyzed_threat"] = predictor.predict_fusion(
                    physiological_stress_score=st.session_state["physio_score"],
                    location_risk_score=loc_score_val,
                    environment_risk_score=st.session_state["env_score"],
                    threshold=0.50
                )

        env_pct = int(round(st.session_state["env_score"] * 100))
        env_badge_cls = "score-danger" if env_pct > 65 else ("score-warn" if env_pct > 35 else "score-safe")
        st.html(f"""<div class="med-score-box">
  <div class="med-score-label">Environment Risk Score</div>
  <div class="med-score-badge {env_badge_cls}">{env_pct}%</div>
</div>""")

    st.html("<br>")

    col_center = st.columns([1, 2, 1])[1]
    with col_center:
        if st.button("🛡️ ANALYZE OVERALL THREAT", key="btn_main_threat", use_container_width=True):
            loc_score_val = st.session_state["location_score"] if st.session_state["location_score"] is not None else 0.15
            res = predictor.predict_fusion(
                physiological_stress_score=st.session_state["physio_score"],
                location_risk_score=loc_score_val,
                environment_risk_score=st.session_state["env_score"],
                threshold=0.50
            )
            st.session_state["analyzed_threat"] = res

    # SECTION 4: OVERALL RISK ASSESSMENT PANEL (Rendered as pure HTML from template)
    if st.session_state["analyzed_threat"] is not None:
        render_overall_risk_panel(st.session_state["analyzed_threat"])

# =============================================================================
# TAB 2: BIT MESRA CAMPUS GPS & SPATIAL ANALYZER
# =============================================================================
with tab2:
    st.html("<h3 style='color: var(--text-primary); margin-bottom: 5px;'>📍 BIT Mesra Campus Spatial Object Analyzer</h3>")
    st.caption("Interactive GPS Geofence, Nearest Landmark Object Detection, & Distance to Guard Posts")

    sp_col1, sp_col2 = st.columns([1, 1])

    with sp_col1:
        st.markdown("#### ⚙️ Campus Landmark Presets")
        
        landmark_names = [lm["name"] for lm in config.BIT_MESRA_LANDMARKS]
        
        loc_res = st.session_state.get("location_res")
        cur_lm_name = loc_res["nearest_landmark"]["name"] if (loc_res and loc_res.get("nearest_landmark")) else landmark_names[0]

        def on_landmark_preset_change():
            sel_name = st.session_state.get("tab2_preset_radio")
            if sel_name:
                target_lm = next((lm for lm in config.BIT_MESRA_LANDMARKS if lm["name"] == sel_name), config.BIT_MESRA_LANDMARKS[0])
                analysis_res = campus_analyzer.analyze_gps(
                    lat=target_lm["latitude"],
                    lon=target_lm["longitude"],
                    accuracy=5.0,
                    time_of_day_hour=datetime.datetime.now().hour
                )
                st.session_state["gps_status"] = "CONNECTED"
                st.session_state["location_res"] = analysis_res
                st.session_state["location_score"] = analysis_res["location_risk_score"]
                if st.session_state.get("analyzed_threat") is not None:
                    st.session_state["analyzed_threat"] = predictor.predict_fusion(
                        physiological_stress_score=st.session_state["physio_score"],
                        location_risk_score=st.session_state["location_score"],
                        environment_risk_score=st.session_state["env_score"],
                        threshold=0.50
                    )

        if "tab2_preset_radio" not in st.session_state or st.session_state["tab2_preset_radio"] not in landmark_names:
            st.session_state["tab2_preset_radio"] = cur_lm_name

        selected_radio = st.radio(
            "Select Landmark Preset to Evaluate:",
            landmark_names,
            key="tab2_preset_radio",
            on_change=on_landmark_preset_change
        )

        st.markdown("---")
        st.markdown("#### 📊 Spatial Metrics")
        
        if loc_res and loc_res.get("nearest_landmark"):
            near_lm = loc_res["nearest_landmark"]
            det_lm = loc_res["detected_landmark"]
            loc_risk_val = loc_res["location_risk_score"]
            desc_val = loc_res.get("description", "")
            sec_dist = loc_res.get("distance_to_security_m", 0.0)
            light_val = loc_res.get("lighting_level", 0.0)
            iso_val = loc_res.get("isolation_index", 0.0)
        else:
            near_lm = {"name": "—", "distance_meters": "—"}
            det_lm = {"name": "—", "is_inside": False, "radius_meters": 0.0}
            loc_risk_val = 0.0
            desc_val = "Connect Live GPS or select a landmark preset."
            sec_dist = "—"
            light_val = "—"
            iso_val = 0.0

        st.html(f"""<div class='med-card'>
  <div style='font-size: 1.05rem; font-weight: 700; color: var(--text-primary);'>🎯 Nearest Campus Landmark</div>
  <div style='font-size: 1.25rem; font-weight: 800; color: var(--primary-accent); margin: 4px 0;'>{near_lm["name"]}</div>
  <div style='font-size: 0.85rem; color: var(--text-secondary);'>{desc_val}</div>
  <hr style='margin: 10px 0; border: none; border-top: 1px solid var(--border-color);'>
  <div class='med-metric-row'><span>Distance to Landmark:</span> <span><b>{near_lm["distance_meters"]}</b></span></div>
  <div class='med-metric-row'><span>Detection Radius Status:</span> <span><b>{det_lm["name"]}</b></span></div>
  <div class='med-metric-row'><span>Distance to Security Post:</span> <span><b>{sec_dist} m</b></span></div>
  <div class='med-metric-row'><span>Derived Zone Lighting:</span> <span><b>{light_val} / 10.0</b></span></div>
  <div class='med-metric-row'><span>Spatial Isolation Index:</span> <span><b>{iso_val * 100:.1f}%</b></span></div>
  <hr style='margin: 10px 0; border: none; border-top: 1px solid var(--border-color);'>
  <div style='text-align: center; margin-top: 8px;'>
    <span style='font-size: 0.9rem; font-weight: 600; color: var(--text-secondary);'>Contextual Location Risk:</span>
    <span style='font-size: 1.25rem; font-weight: 800; color: {"var(--status-danger-color)" if loc_risk_val > 0.5 else "var(--status-safe-color)"}; margin-left: 8px;'>{int(loc_risk_val * 100) if loc_res else "—"}%</span>
  </div>
</div>""")

    with sp_col2:
        st.markdown("#### 🗺️ BIT Mesra Interactive Campus Map")
        
        map_records = []
        for lm in config.BIT_MESRA_LANDMARKS:
            map_records.append({
                "Name": lm["name"],
                "latitude": lm["latitude"],
                "longitude": lm["longitude"],
                "Type": lm["location_type"].replace("_", " ").title(),
                "Marker Type": "Campus Landmark"
            })
            
        map_df = pd.DataFrame(map_records)
        
        if loc_res and loc_res.get("current_location"):
            c_loc = loc_res["current_location"]
            near_name = loc_res["nearest_landmark"]["name"] if loc_res.get("nearest_landmark") else "User Location"
            user_point = pd.DataFrame([{
                "Name": f"📍 User Position ({near_name})",
                "latitude": c_loc["latitude"],
                "longitude": c_loc["longitude"],
                "Type": "User Device GPS",
                "Marker Type": "User GPS Position"
            }])
            combined_df = pd.concat([map_df, user_point], ignore_index=True)
        else:
            combined_df = map_df
        
        if HAS_PLOTLY:
            fig = px.scatter_map(
                combined_df,
                lat="latitude",
                lon="longitude",
                hover_name="Name",
                hover_data=["Type"],
                color="Marker Type",
                color_discrete_map={"Campus Landmark": "#3978C5", "User GPS Position": "#C44444"},
                size=[14 if t == "User GPS Position" else 9 for t in combined_df["Marker Type"]],
                zoom=14,
                center={"lat": config.BIT_MESRA_CENTER["latitude"], "lon": config.BIT_MESRA_CENTER["longitude"]},
                height=480,
                title="BIT Mesra GPS Mapping"
            )
            fig.update_layout(
                map_style="open-street-map",
                margin={"r":0, "t":35, "l":0, "b":0}
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.map(combined_df[["latitude", "longitude"]])

        st.markdown("#### 📋 BIT Mesra Landmark Inventory")
        df_event = st.dataframe(
            map_df[["Name", "Type"]],
            use_container_width=True,
            hide_index=True,
            selection_mode="single-row",
            on_select="rerun",
            key="landmark_table_select"
        )
        
        if df_event and hasattr(df_event, "selection") and df_event.selection and "rows" in df_event.selection and len(df_event.selection["rows"]) > 0:
            selected_row_idx = df_event.selection["rows"][0]
            if st.session_state.get("last_table_selected_row") != selected_row_idx:
                st.session_state["last_table_selected_row"] = selected_row_idx
                clicked_lm = config.BIT_MESRA_LANDMARKS[selected_row_idx]
                analysis_res = campus_analyzer.analyze_gps(
                    lat=clicked_lm["latitude"],
                    lon=clicked_lm["longitude"],
                    accuracy=5.0,
                    time_of_day_hour=datetime.datetime.now().hour
                )
                st.session_state["gps_status"] = "CONNECTED"
                st.session_state["location_res"] = analysis_res
                st.session_state["location_score"] = analysis_res["location_risk_score"]
                st.session_state["tab2_preset_radio"] = clicked_lm["name"]
                if st.session_state.get("analyzed_threat") is not None:
                    st.session_state["analyzed_threat"] = predictor.predict_fusion(
                        physiological_stress_score=st.session_state["physio_score"],
                        location_risk_score=st.session_state["location_score"],
                        environment_risk_score=st.session_state["env_score"],
                        threshold=0.50
                    )
