import os
import sys
import html
import datetime
import pandas as pd
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from pathlib import Path

# Declare custom GPS Sensor Component
_gps_component_dir = Path(__file__).parent / "components" / "gps"
if _gps_component_dir.exists():
    gps_sensor = components.declare_component("gps_sensor", path=str(_gps_component_dir))
else:
    gps_sensor = None

try:
    import plotly.express as px
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

# Ensure project root is prioritized in sys.path
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
elif sys.path[0] != _project_root:
    sys.path.remove(_project_root)
    sys.path.insert(0, _project_root)

import config
from src.fusion.predict import MultimodalThreatPredictor
from src.location.campus_map import CampusMapAnalyzer
from src.environment.model import SUPPORTED_LOCATIONS
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

# Client-side mobile hardening script to prevent text carets, accidental contentEditable, and mobile keyboard triggering on selectboxes
def inject_mobile_protections():
    protection_js = """<script>
(function() {
  function enforceReadOnly() {
    // 1. Ensure document is never in designMode or contentEditable
    try {
      if (document.designMode && document.designMode.toLowerCase() === 'on') {
        document.designMode = 'off';
      }
      if (document.body && document.body.isContentEditable) {
        document.body.contentEditable = 'false';
      }
    } catch(e) {}

    // 2. Ensure selectbox inputs are read-only to prevent Android Chrome soft-keyboard / text carets
    const selectInputs = document.querySelectorAll('div[data-baseweb="select"] input');
    selectInputs.forEach(function(inp) {
      if (!inp.readOnly) inp.readOnly = true;
      if (inp.getAttribute('readonly') !== 'readonly') inp.setAttribute('readonly', 'readonly');
      if (inp.getAttribute('inputmode') !== 'none') inp.setAttribute('inputmode', 'none');
      if (inp.getAttribute('contenteditable') !== 'false') inp.setAttribute('contenteditable', 'false');

      if (!inp._protectionAttached) {
        inp._protectionAttached = true;
        inp.addEventListener('beforeinput', function(e) {
          e.preventDefault();
        });
        inp.addEventListener('keydown', function(e) {
          const allowed = ['Tab', 'Enter', 'Escape', 'ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight'];
          if (allowed.includes(e.key) || [9, 13, 27, 37, 38, 39, 40].includes(e.keyCode)) {
            return;
          }
          e.preventDefault();
        });
        inp.addEventListener('touchstart', function(e) {
          this.readOnly = true;
          this.setAttribute('inputmode', 'none');
        }, { passive: true });
      }
    });

    // 3. Ensure dashboard cards, headings, and static metrics remain strictly read-only
    const staticContainers = document.querySelectorAll(
      '.med-header, .med-card, .med-overall-panel, .med-metric-row, .med-score-box, ' +
      '.med-breakdown-box, .med-rationale-note, [data-testid="stMarkdownContainer"], ' +
      '[data-testid="stText"], [data-testid="stMetric"]'
    );
    staticContainers.forEach(function(el) {
      if (el.getAttribute('contenteditable') !== 'false') {
        el.setAttribute('contenteditable', 'false');
      }
      if (el.isContentEditable) {
        el.contentEditable = 'false';
      }
    });
  }

  enforceReadOnly();
  if (!window._medObserverAttached) {
    window._medObserverAttached = true;
    const observer = new MutationObserver(function() {
      enforceReadOnly();
    });
    observer.observe(document.body, { childList: true, subtree: true });
  }
})();
</script>"""
    st.html(protection_js, unsafe_allow_javascript=True)

inject_mobile_protections()

def render_header():
    header_html = """<div class="med-header">
  <div class="med-header-title-row">
    <div>
      <h1 class="med-title">Multimodal Threat Prediction Engine</h1>
      <p class="med-subtitle">Level 1 Stacking Ensemble Fusion (Physiological · Geospatial · Environmental)</p>
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
    env_norm = res.get("environment_risk_score")

    phys_pct = phys_norm * 100.0
    loc_pct = loc_norm * 100.0
    env_pct = (float(env_norm) * 100.0) if env_norm is not None else None

    fusion_mode = res.get("fusion_mode", "THREE_BRANCH_FUSION")
    is_stacker_trained = bool(res.get("is_meta_learner_trained", False))
    status = str(res.get("context_status", res.get("context_classification", "STANDARD CONTEXT")))
    explanation_txt = html.escape(str(res.get("explanation", "")))

    if status == "HIGH CONTEXT":
        status_badge_cls = "score-danger"
    elif status == "ELEVATED CONTEXT":
        status_badge_cls = "score-warn"
    else:
        status_badge_cls = "score-safe"

    # =========================================================================
    # CASE 1: PARTIAL DUAL-BRANCH CONTEXT INDEX (ENV MISSING / UNKNOWN)
    # =========================================================================
    if fusion_mode == "PARTIAL_DUAL_BRANCH" or res.get("experimental_context_index") is None:
        partial_pct = float(res.get("partial_dual_branch_index_pct", res.get("threat_score_pct", 0.0)))
        contrib_p_partial = phys_pct * (0.40 / 0.65)
        contrib_l_partial = loc_pct * (0.25 / 0.65)

        rendered = f"""<div class="med-overall-panel">
  <div class="med-overall-header">
    Overall Context Assessment — Partial Dual-Branch Context Index (Fallback Mode)
  </div>
  <div class="med-overall-content">
    <div class="med-score-hero">
      <div class="med-score-hero-title">PARTIAL DUAL-BRANCH CONTEXT INDEX (PHYSIOLOGICAL + SPATIAL ONLY)</div>
      <div class="med-score-hero-num">{partial_pct:.2f}%</div>
      <div style="margin: 8px 0; font-size: 0.88rem; font-weight: 600; color: var(--text-secondary);">
        Context Classification: <span class="med-score-badge {status_badge_cls}" style="font-size: 0.85rem; padding: 2px 8px; vertical-align: middle;">{status}</span>
      </div>
      <div style="margin-top: 6px;">
        <span class="med-provenance-pill pill-rule">Meta-Learner: NOT TRAINED</span>
      </div>
      <div style="margin-top: 12px; font-size: 0.78rem; color: var(--text-secondary); line-height: 1.4; font-style: italic;">
        This partial index evaluates physiological stress and spatial location only. Environmental context is UNKNOWN.
      </div>
    </div>
    <div class="med-breakdown-box">
      <div class="med-alert-warn" style="margin-bottom: 12px; font-size: 0.88rem; line-height: 1.45;">
        ⚠️ <b>COMPLETE THREE-BRANCH CONTEXT INDEX UNAVAILABLE (ENVIRONMENTAL UNKNOWN):</b> Environmental context inputs are missing or unverified. No neutral values are silently substituted.
      </div>
      <div style="font-weight: 700; font-size: 0.95rem; color: var(--text-primary); margin-bottom: 8px;">
        Partial Dual-Branch Re-Normalized Weighting Breakdown
      </div>
      <div class="med-breakdown-item">
        <span>Physiological Stress (Re-normalized 61.54% Weight):</span>
        <span><b>{phys_pct:.1f}% × 0.6154 = {contrib_p_partial:.2f}%</b></span>
      </div>
      <div class="med-breakdown-item">
        <span>Location Context Risk (Re-normalized 38.46% Weight):</span>
        <span><b>{loc_pct:.1f}% × 0.3846 = {contrib_l_partial:.2f}%</b></span>
      </div>
      <div class="med-breakdown-item" style="color: var(--text-secondary); opacity: 0.7;">
        <span>Environmental Context (Excluded):</span>
        <span><b>UNKNOWN (Excluded from index calculation)</b></span>
      </div>
      <div style="margin-top: 10px; font-size: 0.83rem; color: var(--text-secondary); font-style: italic;">
        ℹ️ <b>Formula:</b> <code>R_partial = (0.40 × P + 0.25 × L) / 0.65</code>. Situational illumination and crowd density are omitted.
      </div>
      <div class="med-rationale-note">
        <b>Decision Rationale:</b> {explanation_txt}
      </div>
    </div>
  </div>
</div>"""
        st.html(rendered)
        return

    # =========================================================================
    # CASE 2: COMPLETE THREE-BRANCH CONTEXT ASSESSMENT (ALL 3 PRESENT)
    # =========================================================================
    final_prob = float(res.get("experimental_context_index", res.get("threat_probability", 0.0)))
    final_pct = float(res.get("experimental_context_index_pct", res.get("threat_score_pct", final_prob * 100.0)))

    contrib_p = phys_pct * 0.40
    contrib_l = loc_pct * 0.25
    contrib_e = (env_pct * 0.35) if env_pct is not None else 0.0

    if is_stacker_trained:
        # Rendered only when an independently verified production empirical model has been trained
        meta_explain = res.get("stacking_explainability") or {}
        b0 = float(meta_explain.get("intercept_beta0", 0.0))
        coefs = meta_explain.get("coefficients", {})
        coef_items = " | ".join([f"{k}: β={v:+.3f}" for k, v in coefs.items()])
        rendered = f"""<div class="med-overall-panel">
  <div class="med-overall-header">
    Overall Context Assessment — Level 1 Stacking Ensemble (Empirical)
  </div>
  <div class="med-overall-content">
    <div class="med-score-hero">
      <div class="med-score-hero-title">EXPERIMENTAL WEIGHTED CONTEXT INDEX</div>
      <div class="med-score-hero-num">{final_pct:.2f}%</div>
      <div style="margin: 8px 0; font-size: 0.88rem; font-weight: 600; color: var(--text-secondary);">
        Context Classification: <span class="med-score-badge {status_badge_cls}" style="font-size: 0.85rem; padding: 2px 8px; vertical-align: middle;">{status}</span>
      </div>
      <div style="margin-top: 6px;">
        <span class="med-provenance-pill pill-stacking">⚙️ Meta-Learner: TRAINED ML (Production-Eligible)</span>
      </div>
      <div style="margin-top: 12px; font-size: 0.78rem; color: var(--text-secondary); line-height: 1.4; font-style: italic;">
        This index represents a heuristic combination of physiological, spatial and environmental context scores. It is not a validated threat prediction.
      </div>
    </div>
    <div class="med-breakdown-box">
      <div style="font-weight: 700; font-size: 0.95rem; color: var(--text-primary); margin-bottom: 8px;">
        Learned Stacking Formulation: <code>logit(P) = β₀ + β₁P₁ + β₂P₂ + β₃P₃</code>
      </div>
      <div class="med-metric-row" style="font-size: 0.84rem; background: #F8FAFC; padding: 6px 8px; border-radius: 4px; margin-bottom: 6px;">
        <span><b>Learned Coefficients:</b></span>
        <span>β₀={b0:+.3f} | {coef_items}</span>
      </div>
      <div class="med-rationale-note">
        <b>Decision Rationale:</b> {explanation_txt}
      </div>
    </div>
  </div>
</div>"""
    else:
        # Production standard state: Untrained meta-learner with Experimental Weighted Context Index
        rendered = f"""<div class="med-overall-panel">
  <div class="med-overall-header">
    Overall Context Assessment — Experimental Weighted Context Index
  </div>
  <div class="med-overall-content">
    <div class="med-score-hero">
      <div class="med-score-hero-title">EXPERIMENTAL WEIGHTED CONTEXT INDEX</div>
      <div class="med-score-hero-num">{final_pct:.2f}%</div>
      <div style="margin: 8px 0; font-size: 0.88rem; font-weight: 600; color: var(--text-secondary);">
        Context Classification: <span class="med-score-badge {status_badge_cls}" style="font-size: 0.85rem; padding: 2px 8px; vertical-align: middle;">{status}</span>
      </div>
      <div style="margin-top: 6px;">
        <span class="med-provenance-pill pill-rule">Meta-Learner: NOT TRAINED</span>
      </div>
      <div style="margin-top: 12px; font-size: 0.78rem; color: var(--text-secondary); line-height: 1.4; font-style: italic;">
        This index represents a heuristic combination of physiological, spatial and environmental context scores. It is not a validated threat prediction.
      </div>
    </div>
    <div class="med-breakdown-box">
      <div class="med-alert-warn" style="margin-bottom: 12px; font-size: 0.88rem; line-height: 1.45;">
        ⚠️ <b>Important Notice:</b> A validated multimodal threat-prediction model is not currently available because synchronized observations and independently verified outcome labels have not yet been collected.
      </div>
      <div style="font-weight: 700; font-size: 0.95rem; color: var(--text-primary); margin-bottom: 8px;">
        Heuristic Weighting Breakdown (Reference Baseline)
      </div>
      <div class="med-breakdown-item">
        <span>Physiological Stress (40% Weight):</span>
        <span><b>{phys_pct:.1f}% × 0.40 = {contrib_p:.2f}%</b></span>
      </div>
      <div class="med-breakdown-item">
        <span>Location Context Risk (25% Weight):</span>
        <span><b>{loc_pct:.1f}% × 0.25 = {contrib_l:.2f}%</b></span>
      </div>
      <div class="med-breakdown-item">
        <span>Environmental Context Risk (35% Weight):</span>
        <span><b>{env_pct:.1f}% × 0.35 = {contrib_e:.2f}%</b></span>
      </div>
      <div style="margin-top: 10px; font-size: 0.83rem; color: var(--text-secondary); font-style: italic;">
        ℹ️ <b>Clarification:</b> This value describes contextual inputs only and does not establish whether a person or location is safe or unsafe.
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
    """Initializes and caches the unified MultimodalThreatPredictor singleton (v2.1)."""
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
if "location_is_cached" not in st.session_state:
    st.session_state["location_is_cached"] = False
if "gps_error_msg" not in st.session_state:
    st.session_state["gps_error_msg"] = None
if "gps_force_trigger" not in st.session_state:
    st.session_state["gps_force_trigger"] = 0
if "env_score" not in st.session_state:
    st.session_state["env_score"] = 0.10
if "analyzed_threat" not in st.session_state:
    st.session_state["analyzed_threat"] = None
if "gps_query_counter" not in st.session_state:
    st.session_state["gps_query_counter"] = 0
if "last_table_selected_row" not in st.session_state:
    st.session_state["last_table_selected_row"] = None
if "client_hour" not in st.session_state:
    st.session_state["client_hour"] = None
if "client_time_str" not in st.session_state:
    st.session_state["client_time_str"] = None

landmark_names = [lm["name"] for lm in config.BIT_MESRA_LANDMARKS]
if "selected_landmark_name" not in st.session_state or st.session_state["selected_landmark_name"] not in landmark_names:
    st.session_state["selected_landmark_name"] = landmark_names[0]

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
  <div>
    <span class="med-card-title">1. Physiological Analysis</span>
    <span class="med-card-subtitle">WESAD Wearable Signals</span>
  </div>
  <span class="med-provenance-pill pill-ml">⚡ WESAD-trained ML model — Physiological Stress Detection</span>
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
  <div class="med-score-label">Physiological Stress Probability</div>
  <div class="med-score-badge {badge_cls}">{phys_pct}%</div>
</div>""")

    # CARD 2 — CURRENT LOCATION (LIVE BROWSER GPS)
    with col2:
        st.html("""<div class="med-card-header">
  <div>
    <span class="med-card-title">2. Current Location</span>
    <span class="med-card-subtitle">Browser Geolocation</span>
  </div>
  <span class="med-provenance-pill pill-rule">📐 Deterministic Geospatial Context Engine</span>
</div>""")

        # Render custom reliable GPS sensor component
        if gps_sensor is not None:
            gps_payload = gps_sensor(
                key="live_gps_sensor",
                force_request_id=st.session_state.get("gps_force_trigger", 0),
                initial_status=st.session_state.get("gps_status", "OFF")
            )
        else:
            gps_payload = None

        if gps_payload and isinstance(gps_payload, dict):
            p_status = gps_payload.get("status")
            if "client_hour" in gps_payload and gps_payload["client_hour"] is not None:
                st.session_state["client_hour"] = float(gps_payload["client_hour"])
            if "client_time_str" in gps_payload and gps_payload["client_time_str"] is not None:
                st.session_state["client_time_str"] = str(gps_payload["client_time_str"])

            if p_status == "SUCCESS" and "coords" in gps_payload:
                coords = gps_payload["coords"]
                r_lat = float(coords["latitude"])
                r_lon = float(coords["longitude"])
                r_acc = float(coords.get("accuracy", 10.0))
                r_ts = gps_payload.get("timestamp")
                is_cached = bool(gps_payload.get("is_cached", False))

                now_campus = config.get_campus_now()
                calc_hour = st.session_state.get("client_hour") or (now_campus.hour + now_campus.minute / 60.0)

                analysis_res = campus_analyzer.analyze_gps(
                    lat=r_lat,
                    lon=r_lon,
                    accuracy=r_acc,
                    timestamp=r_ts,
                    time_of_day_hour=calc_hour
                )

                st.session_state["gps_status"] = analysis_res["status"]
                st.session_state["location_res"] = analysis_res
                st.session_state["location_score"] = analysis_res["location_risk_score"]
                st.session_state["location_is_cached"] = is_cached
                st.session_state["gps_error_msg"] = None

                if st.session_state.get("analyzed_threat") is not None:
                    st.session_state["analyzed_threat"] = predictor.predict_fusion(
                        physiological_stress_score=st.session_state["physio_score"],
                        location_risk_score=st.session_state["location_score"],
                        environment_risk_score=st.session_state["env_score"],
                        threshold=0.50
                    )
            elif p_status in ["PERMISSION_DENIED", "POSITION_UNAVAILABLE", "TIMEOUT", "ERROR", "NOT_SUPPORTED"]:
                st.session_state["gps_status"] = p_status
                st.session_state["location_res"] = None
                st.session_state["location_score"] = None
                st.session_state["gps_error_msg"] = gps_payload.get("message", "GPS acquisition failed.")

        # Provide working Python Retry button if GPS encountered error/timeout
        if st.session_state["gps_status"] in ["PERMISSION_DENIED", "POSITION_UNAVAILABLE", "TIMEOUT", "ERROR"]:
            if st.button("🔄 [ Retry GPS Fix ]", key="btn_retry_gps", use_container_width=True):
                st.session_state["gps_force_trigger"] += 1
                st.session_state["gps_status"] = "OFF"
                st.session_state["gps_error_msg"] = None
                st.rerun()

        loc_res = st.session_state.get("location_res")
        has_valid_gps = (loc_res is not None and loc_res.get("nearest_landmark") is not None)

        if has_valid_gps:
            c_loc = loc_res["current_location"]
            n_lm = loc_res["nearest_landmark"]
            acc_val = float(c_loc['accuracy_meters'])
            lat_disp = f"{c_loc['latitude']:.6f}"
            lon_disp = f"{c_loc['longitude']:.6f}"

            is_low_acc = acc_val > 100.0
            acc_note = " (⚠️ Low Accuracy)" if is_low_acc else ""
            acc_disp = f"±{int(acc_val)} m{acc_note}"

            fix_freshness = "Cached Fix (Fresh)" if st.session_state.get("location_is_cached") else "Live GPS Fix"
            near_lm_disp = f"{n_lm['name']} (Proximity Approximate)" if is_low_acc else n_lm["name"]
            dist_disp = f"{n_lm['distance_meters']} m"
            score_disp = f"{int(loc_res['location_risk_score'] * 100)}%"
            score_val = loc_res["location_risk_score"]
        else:
            lat_disp = "—"
            lon_disp = "—"
            acc_disp = "—"
            fix_freshness = "—"
            near_lm_disp = "—"
            dist_disp = "—"
            score_disp = "—"
            score_val = 0.0
            is_low_acc = False

        st.html(f"""<div style='margin-top: 10px;'>
  <div class='med-metric-row'><span class='med-metric-label'>Latitude</span> <span class='med-metric-value'>{lat_disp}</span></div>
  <div class='med-metric-row'><span class='med-metric-label'>Longitude</span> <span class='med-metric-value'>{lon_disp}</span></div>
  <div class='med-metric-row'><span class='med-metric-label'>Accuracy</span> <span class='med-metric-value'>{acc_disp}</span></div>
  <div class='med-metric-row'><span class='med-metric-label'>Fix Freshness</span> <span class='med-metric-value'>{fix_freshness}</span></div>
  <div class='med-metric-row'><span class='med-metric-label'>Nearest Landmark</span> <span class='med-metric-value med-metric-value-accent'>{near_lm_disp}</span></div>
  <div class='med-metric-row'><span class='med-metric-label'>Distance</span> <span class='med-metric-value'>{dist_disp}</span></div>
</div>""")

        if has_valid_gps and is_low_acc:
            st.html(f"""<div class='med-alert-warn'>
⚠️ Reported accuracy is ±{int(acc_val)}m. The nearest campus landmark is approximate and subject to GPS drift.
</div>""")

        if st.session_state.get("gps_error_msg"):
            st.html(f"""<div class='med-alert-error'>
⚠️ {st.session_state["gps_error_msg"]}
</div>""")

        badge_cls = "score-danger" if score_val > 0.65 else ("score-warn" if score_val > 0.35 else "score-safe")
        st.html(f"""<div class="med-score-box">
  <div class="med-score-label">Location Vulnerability Score</div>
  <div class="med-score-badge {badge_cls}">{score_disp}</div>
</div>""")

    # CARD 3 — ENVIRONMENTAL CONTEXT (AUTOMATIC AI PREDICTION)
    with col3:
        st.html("""<div class="med-card-header">
  <div>
    <span class="med-card-title">3. Environmental Context</span>
    <span class="med-card-subtitle">AI-Based Situational Prediction</span>
  </div>
  <span class="med-provenance-pill pill-ml">⚡ Synthetic-Trained ML Prediction Engine</span>
</div>""")

        # 1. Automatic Time and Day Detection (Asia/Kolkata)
        try:
            if hasattr(predictor, "get_current_ist_time"):
                ist_time_info = predictor.get_current_ist_time()
            elif hasattr(predictor, "env_engine") and predictor.env_engine is not None and hasattr(predictor.env_engine, "get_current_ist_time"):
                ist_time_info = predictor.env_engine.get_current_ist_time()
            else:
                ist_time_info = config.get_current_ist_time()
        except Exception:
            ist_time_info = config.get_current_ist_time()

        curr_time_disp = ist_time_info.get("time_str", "12:00 PM IST")
        curr_day_disp = ist_time_info.get("day_of_week", "Monday")
        is_wknd = ist_time_info.get("is_weekend", 0)
        hour_val = ist_time_info.get("hour_float", 12.0)
        wknd_tag = "Weekend" if is_wknd else "Weekday"

        # 2. Location Resolution (GPS vs Ambiguity vs Manual Fallback)
        loc_res = st.session_state.get("location_res")
        has_gps_landmark = (loc_res is not None and loc_res.get("nearest_landmark") is not None)
        gps_resolved_name = loc_res["nearest_landmark"]["name"] if has_gps_landmark else None

        gps_resolve_info = None
        if loc_res is not None and loc_res.get("current_location") is not None and hasattr(predictor, "env_engine") and predictor.env_engine is not None:
            c_loc = loc_res["current_location"]
            gps_resolve_info = predictor.env_engine.resolve_campus_location(
                lat=c_loc.get("latitude"),
                lon=c_loc.get("longitude"),
                accuracy=c_loc.get("accuracy_meters"),
                max_geofence_meters=350.0
            )

        is_ambiguous = (gps_resolve_info is not None and gps_resolve_info.get("status") == "AMBIGUOUS_LOCATION")
        is_outside_geofence = (gps_resolve_info is not None and gps_resolve_info.get("status") == "OUTSIDE_CAMPUS_GEOFENCE")

        if is_ambiguous:
            cands = gps_resolve_info.get("ambiguous_candidates", [])
            cand_str = ", ".join(cands)
            st.html(f"""<div class='med-alert-warn' style='font-size:0.78rem; margin-bottom:8px;'>
⚠️ <strong>AMBIGUOUS GPS LOCATION DETECTED:</strong> Position is within accuracy margin between: <em>{cand_str}</em>. Please manually confirm your campus location below.
</div>""")
            default_cand = cands[0] if cands and cands[0] in SUPPORTED_LOCATIONS else SUPPORTED_LOCATIONS[0]
            default_idx = SUPPORTED_LOCATIONS.index(default_cand)
            active_env_location = st.selectbox("Confirm Campus Landmark (Ambiguous GPS)", SUPPORTED_LOCATIONS, index=default_idx, key="sel_ambig_env_loc")
            loc_origin = "Manual Confirmation (Ambiguous GPS)"

        elif is_outside_geofence:
            dist_m = gps_resolve_info.get("distance_meters", "—")
            st.html(f"""<div class='med-alert-warn' style='font-size:0.78rem; margin-bottom:8px;'>
⚠️ <strong>OUTSIDE CAMPUS GEOFENCE:</strong> Position is {dist_m}m away, exceeding 350m campus boundary. Please manually select a campus location below.
</div>""")
            active_env_location = st.selectbox("Campus Landmark (Manual Fallback)", SUPPORTED_LOCATIONS, index=0, key="sel_outside_env_loc")
            loc_origin = "Manual Selection (Outside Geofence)"

        elif has_gps_landmark and gps_resolve_info is not None and gps_resolve_info.get("status") == "RESOLVED_GPS":
            resolved_gps_loc = gps_resolve_info.get("resolved_location") or gps_resolved_name
            use_manual_loc = st.checkbox("⚙️ Override GPS Location", value=False, key="cb_manual_env_override")
            if use_manual_loc:
                default_idx = SUPPORTED_LOCATIONS.index(resolved_gps_loc) if resolved_gps_loc in SUPPORTED_LOCATIONS else 0
                active_env_location = st.selectbox("Campus Landmark (Manual Override)", SUPPORTED_LOCATIONS, index=default_idx, key="sel_manual_env_loc")
                loc_origin = "Manual Selection (Override)"
            else:
                active_env_location = resolved_gps_loc
                loc_origin = "Live GPS Resolved"
        elif has_gps_landmark:
            use_manual_loc = st.checkbox("⚙️ Override GPS Location", value=False, key="cb_manual_env_override")
            if use_manual_loc:
                default_idx = SUPPORTED_LOCATIONS.index(gps_resolved_name) if gps_resolved_name in SUPPORTED_LOCATIONS else 0
                active_env_location = st.selectbox("Campus Landmark (Manual Override)", SUPPORTED_LOCATIONS, index=default_idx, key="sel_manual_env_loc")
                loc_origin = "Manual Selection (Override)"
            else:
                active_env_location = gps_resolved_name
                loc_origin = "Live GPS Resolved"
        else:
            st.caption("📍 GPS unavailable or manual mode. Select campus location:")
            active_env_location = st.selectbox("Campus Landmark (Manual Fallback)", SUPPORTED_LOCATIONS, index=0, key="sel_manual_env_loc_fallback")
            loc_origin = "Manual Selection (GPS Unavailable)"

        # 3. Automatic AI Environmental Inference
        try:
            env_pred = predictor.predict_environment_context(
                location=active_env_location,
                hour=hour_val,
                day_of_week=curr_day_disp,
                is_weekend=is_wknd
            )
        except Exception as e:
            env_pred = {
                "status": "UNAVAILABLE",
                "error": str(e),
                "message": f"Environmental inference pipeline error: {str(e)}",
                "crowd_prediction": {"category": "UNKNOWN", "confidence_pct": 0.0, "model_name": "Unavailable"},
                "lighting_prediction": {"category": "UNKNOWN", "confidence_pct": 0.0, "model_name": "Unavailable", "lux_estimate": None},
                "environmental_context_score": None,
                "calibrated_environmental_score": None
            }

        if env_pred.get("status") == "SUCCESS" and env_pred.get("environmental_context_score") is not None:
            st.session_state["env_score"] = env_pred["environmental_context_score"]

            crowd_res = env_pred["crowd_prediction"]
            light_res = env_pred["lighting_prediction"]
            crowd_cat = crowd_res["category"]
            crowd_conf = crowd_res["confidence_pct"]
            crowd_model = crowd_res["model_name"]

            light_cat = light_res["category"]
            light_conf = light_res["confidence_pct"]
            light_model = light_res["model_name"]
            lux_val = light_res.get("lux_estimate", "—")

            st.html(f"""<div style='margin-top: 6px;'>
  <div class='med-metric-row'><span class='med-metric-label'>Location</span> <span class='med-metric-value med-metric-value-accent'>{active_env_location}</span></div>
  <div class='med-metric-row'><span class='med-metric-label'>Location Mode</span> <span class='med-metric-value' style='font-size:0.8rem;'>{loc_origin}</span></div>
  <div class='med-metric-row'><span class='med-metric-label'>Time (Asia/Kolkata)</span> <span class='med-metric-value'>{curr_time_disp}</span></div>
  <div class='med-metric-row'><span class='med-metric-label'>Day of Week</span> <span class='med-metric-value'>{curr_day_disp} ({wknd_tag})</span></div>
</div>

<div style='display:flex; gap:8px; margin-top:10px;'>
  <div style='flex:1; background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.08); border-radius:6px; padding:8px;'>
    <div style='font-size:0.75rem; color:#888;'>Crowd Density</div>
    <div style='font-size:1.05rem; font-weight:700; color:#fff;'>{crowd_cat} <span style='font-size:0.75rem; color:#aaa;'>({crowd_conf}%)</span></div>
    <div style='font-size:0.68rem; color:#00bcd4; margin-top:2px;'>{crowd_model}</div>
  </div>
  <div style='flex:1; background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.08); border-radius:6px; padding:8px;'>
    <div style='font-size:0.75rem; color:#888;'>Lighting Condition</div>
    <div style='font-size:1.05rem; font-weight:700; color:#fff;'>{light_cat} <span style='font-size:0.75rem; color:#aaa;'>({light_conf}%)</span></div>
    <div style='font-size:0.68rem; color:#00bcd4; margin-top:2px;'>{light_model} · ~{lux_val} lx</div>
  </div>
</div>

<div class='med-alert-warn' style='font-size:0.75rem; margin-top:8px; padding:6px 10px; line-height:1.3;'>
  ⚠️ <strong>SYNTHETIC-DATA TRAINED MODEL — EXPERIMENTAL PREDICTION</strong><br>
  <strong>NOT LIVE SENSOR MEASUREMENT.</strong> Actual campus crowd and lighting conditions may differ from predictions.
</div>""")
        else:
            st.session_state["env_score"] = None
            crowd_res = env_pred.get("crowd_prediction", {})
            light_res = env_pred.get("lighting_prediction", {})
            crowd_cat = crowd_res.get("category", "UNKNOWN")
            crowd_conf = crowd_res.get("confidence_pct", 0.0)
            light_cat = light_res.get("category", "UNKNOWN")
            light_conf = light_res.get("confidence_pct", 0.0)
            warn_msg = env_pred.get("message") or env_pred.get("warning") or "Environmental models unavailable or unsupported campus location."

            st.html(f"""<div style='margin-top: 6px;'>
  <div class='med-metric-row'><span class='med-metric-label'>Location</span> <span class='med-metric-value med-metric-value-accent'>{active_env_location}</span></div>
  <div class='med-metric-row'><span class='med-metric-label'>Location Mode</span> <span class='med-metric-value' style='font-size:0.8rem;'>{loc_origin}</span></div>
  <div class='med-metric-row'><span class='med-metric-label'>Time (Asia/Kolkata)</span> <span class='med-metric-value'>{curr_time_disp}</span></div>
  <div class='med-metric-row'><span class='med-metric-label'>Day of Week</span> <span class='med-metric-value'>{curr_day_disp} ({wknd_tag})</span></div>
</div>

<div style='display:flex; gap:8px; margin-top:10px;'>
  <div style='flex:1; background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.08); border-radius:6px; padding:8px;'>
    <div style='font-size:0.75rem; color:#888;'>Crowd Density</div>
    <div style='font-size:1.05rem; font-weight:700; color:#aaa;'>{crowd_cat} <span style='font-size:0.75rem; color:#888;'>({crowd_conf:.1f}%)</span></div>
    <div style='font-size:0.68rem; color:#ff9800; margin-top:2px;'>Unavailable</div>
  </div>
  <div style='flex:1; background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.08); border-radius:6px; padding:8px;'>
    <div style='font-size:0.75rem; color:#888;'>Lighting Condition</div>
    <div style='font-size:1.05rem; font-weight:700; color:#aaa;'>{light_cat} <span style='font-size:0.75rem; color:#888;'>({light_conf:.1f}%)</span></div>
    <div style='font-size:0.68rem; color:#ff9800; margin-top:2px;'>Unavailable</div>
  </div>
</div>

<div class='med-alert-warn' style='font-size:0.75rem; margin-top:8px; padding:6px 10px; line-height:1.3;'>
  ⚠️ <strong>CONTEXT STATUS UNKNOWN:</strong> {warn_msg}<br>
  No neutral values are silently substituted. Complete 3-branch index is unavailable. The engine evaluates a Partial Dual-Branch Context Index (Physiological + Spatial only).
</div>""")

        # Auto-update threat fusion if previously analyzed
        if st.session_state.get("analyzed_threat") is not None:
            loc_score_val = st.session_state["location_score"] if st.session_state["location_score"] is not None else 0.15
            try:
                st.session_state["analyzed_threat"] = predictor.predict_fusion(
                    physiological_stress_score=st.session_state["physio_score"],
                    location_risk_score=loc_score_val,
                    environment_risk_score=st.session_state["env_score"],
                    threshold=0.50
                )
            except Exception as e:
                print(f"[app/main.py] Warning: Auto-update fusion failed: {e}", flush=True)

        if st.session_state["env_score"] is not None:
            env_pct = int(round(st.session_state["env_score"] * 100))
            env_badge_cls = "score-danger" if env_pct > 65 else ("score-warn" if env_pct > 35 else "score-safe")
            score_inner = f"<div class='med-score-badge {env_badge_cls}'>{env_pct}%</div>"
            score_lbl = "Environment Vulnerability Score"
        else:
            score_inner = "<div class='med-score-badge score-warn' style='font-size:0.85rem;'>UNKNOWN</div>"
            score_lbl = "Environment Vulnerability Score (Unavailable)"

        st.html(f"""<div class="med-score-box">
  <div class="med-score-label">{score_lbl}</div>
  {score_inner}
</div>""")

    st.html("<br>")

    col_center = st.columns([1, 2, 1])[1]
    with col_center:
        if st.button("🛡️ EVALUATE OVERALL CONTEXT RISK", key="btn_main_threat", use_container_width=True):
            loc_score_val = st.session_state["location_score"] if st.session_state["location_score"] is not None else 0.15
            try:
                res = predictor.predict_fusion(
                    physiological_stress_score=st.session_state["physio_score"],
                    location_risk_score=loc_score_val,
                    environment_risk_score=st.session_state["env_score"],
                    threshold=0.50
                )
                st.session_state["analyzed_threat"] = res
            except Exception as e:
                st.error(f"Fusion Model Evaluation Error: {e}")

    # SECTION 4: OVERALL RISK ASSESSMENT PANEL
    if st.session_state.get("analyzed_threat") is not None:
        try:
            render_overall_risk_panel(st.session_state["analyzed_threat"])
        except Exception as e:
            st.error(f"Error rendering overall context assessment: {e}")

# =============================================================================
# TAB 2: BIT MESRA CAMPUS GPS & SPATIAL ANALYZER
# =============================================================================
with tab2:
    st.html("<h3 style='color: var(--text-primary); margin-bottom: 5px;'>📍 BIT Mesra Campus Spatial Object Analyzer</h3>")
    st.caption("Interactive GPS Geofence, Nearest Landmark Object Detection, & Distance to Guard Posts")

    sp_col1, sp_col2 = st.columns([1, 1])

    def apply_landmark_selection(target_lm):
        if not target_lm:
            return
        now_campus_tab = config.get_campus_now()
        calc_hour = st.session_state.get("client_hour") or (now_campus_tab.hour + now_campus_tab.minute / 60.0)
        analysis_res = campus_analyzer.analyze_gps(
            lat=target_lm["latitude"],
            lon=target_lm["longitude"],
            accuracy=5.0,
            time_of_day_hour=calc_hour
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

    def on_landmark_preset_change():
        sel_name = st.session_state.get("tab2_preset_radio_widget")
        if sel_name:
            st.session_state["selected_landmark_name"] = sel_name
            target_lm = next((lm for lm in config.BIT_MESRA_LANDMARKS if lm["name"] == sel_name), config.BIT_MESRA_LANDMARKS[0])
            apply_landmark_selection(target_lm)

    def on_landmark_table_select():
        table_state = st.session_state.get("landmark_table_select")
        if not table_state:
            return
        selection = getattr(table_state, "selection", None) or (table_state.get("selection") if isinstance(table_state, dict) else None)
        if not selection:
            return
        rows = getattr(selection, "rows", None) or (selection.get("rows") if isinstance(selection, dict) else None)
        if rows and len(rows) > 0:
            selected_row_idx = rows[0]
            if 0 <= selected_row_idx < len(config.BIT_MESRA_LANDMARKS):
                st.session_state["last_table_selected_row"] = selected_row_idx
                clicked_lm = config.BIT_MESRA_LANDMARKS[selected_row_idx]
                st.session_state["selected_landmark_name"] = clicked_lm["name"]
                apply_landmark_selection(clicked_lm)
        else:
            st.session_state["last_table_selected_row"] = None

    with sp_col1:
        st.markdown("#### ⚙️ Campus Landmark Presets")

        loc_res = st.session_state.get("location_res")
        cur_selected = st.session_state.get("selected_landmark_name", landmark_names[0])
        cur_idx = landmark_names.index(cur_selected) if cur_selected in landmark_names else 0

        # Dedicated widget key decoupled from external mutation
        selected_radio = st.radio(
            "Select Landmark Preset to Evaluate:",
            landmark_names,
            index=cur_idx,
            key="tab2_preset_radio_widget",
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
    <span style='font-size: 0.9rem; font-weight: 600; color: var(--text-secondary);'>Contextual Location Vulnerability:</span>
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
        st.dataframe(
            map_df[["Name", "Type"]],
            use_container_width=True,
            hide_index=True,
            selection_mode="single-row",
            on_select=on_landmark_table_select,
            key="landmark_table_select"
        )
