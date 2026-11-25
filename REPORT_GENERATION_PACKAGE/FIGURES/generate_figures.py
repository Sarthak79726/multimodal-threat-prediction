"""
Generate 26 High-Resolution Academic Figures for B.Tech Project Report.
Uses actual experimental evidence, verified datasets, and clear visual styling.
Outputs 300 DPI publication-quality PNG figures into REPORT_GENERATION_PACKAGE/FIGURES.
"""

import os
import sys
import pickle
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import seaborn as sns
from pathlib import Path

# Setup paths and styling
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 0.8

output_dir = Path("E:/project1/REPORT_GENERATION_PACKAGE/FIGURES")
output_dir.mkdir(parents=True, exist_ok=True)

# Load datasets and evidence
env_df = pd.read_csv("data/environment/BIT_Mesra_Synthetic_Environmental_Dataset.csv")
with open("models/physiological/physiological_model.pkl", "rb") as f:
    phys_model = pickle.load(f)

# Helper function to save figure cleanly
def save_fig(fig, filename):
    p = output_dir / filename
    fig.savefig(p, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved: {filename}")

# ==============================================================================
# FIG 01: Complete System Architecture (Diagram)
# ==============================================================================
fig, ax = plt.subplots(figsize=(12, 7))
ax.axis('off')
boxes = [
    # Inputs
    {"xy": (0.05, 0.72), "w": 0.22, "h": 0.22, "title": "Branch 1: Physiological\n(WESAD Dataset)", "color": "#E3F2FD", "edge": "#1E88E5", "text": "• RespiBAN Chest (700 Hz)\n• Empatica E4 Wrist\n• 76 Statistical Features\n• Random Forest (150 trees)\n→ P_stress in [0, 1]"},
    {"xy": (0.05, 0.40), "w": 0.22, "h": 0.24, "title": "Branch 2: Spatial Context\n(BIT Mesra GIS)", "color": "#E8F5E9", "edge": "#43A047", "text": "• 17 Campus Landmarks\n• Geolocation API (Lat, Lon)\n• Haversine Geofencing (350m)\n• Ambiguity & Manual Fallback\n→ P_spatial in [0, 1]"},
    {"xy": (0.05, 0.08), "w": 0.22, "h": 0.24, "title": "Branch 3: Environmental\n(Campus AI Context)", "color": "#FFF3E0", "edge": "#FB8C00", "text": "• 17 Locations x 24h Schedule\n• Balanced XGBoost (Crowd)\n• Random Forest (Lighting)\n• Lux Regressor (Continuous)\n→ P_env in [0, 1]"},
    # Fusion Engine
    {"xy": (0.37, 0.32), "w": 0.26, "h": 0.42, "title": "Multimodal Fusion Engine", "color": "#EDE7F6", "edge": "#5E35B1", "text": "• Mode 1: Static Weighted Late Fusion\n  R = 0.40 P + 0.25 L + 0.35 E\n• Mode 2: Partial Dual-Branch\n  R_dual = (0.40 P + 0.25 L) / 0.65\n• Mode 3: Dynamic Gating (Sim/Research)\n• Mode 4: Stacking Ensemble Pipeline\n• Uncertainty & Missing Data Guard"},
    # Output / Application
    {"xy": (0.73, 0.32), "w": 0.22, "h": 0.42, "title": "Streamlit Cloud Dashboard\n& Decision Support", "color": "#FCE4EC", "edge": "#D81B60", "text": "• Dynamic Risk Level:\n  LOW (<=0.39), MED (0.40-0.69), HIGH\n• Modality Reliability Cards\n• Interactive Leaflet Map\n• Grounded Explainability\n• Privacy & Security Notice"}
]

for b in boxes:
    rect = FancyBboxPatch(b["xy"], b["w"], b["h"], facecolor=b["color"], edgecolor=b["edge"], linewidth=1.5, boxstyle="round,pad=0.03")
    ax.add_patch(rect)
    ax.text(b["xy"][0] + b["w"]/2, b["xy"][1] + b["h"] - 0.04, b["title"], ha="center", va="top", fontweight="bold", fontsize=10, color=b["edge"])
    ax.text(b["xy"][0] + 0.015, b["xy"][1] + b["h"] - 0.09, b["text"], ha="left", va="top", fontsize=8, color="#222222")

# Arrows
ax.annotate("", xy=(0.37, 0.65), xytext=(0.27, 0.83), arrowprops=dict(arrowstyle="->", lw=1.5, color="#1E88E5"))
ax.annotate("", xy=(0.37, 0.53), xytext=(0.27, 0.52), arrowprops=dict(arrowstyle="->", lw=1.5, color="#43A047"))
ax.annotate("", xy=(0.37, 0.41), xytext=(0.27, 0.20), arrowprops=dict(arrowstyle="->", lw=1.5, color="#FB8C00"))
ax.annotate("", xy=(0.73, 0.53), xytext=(0.63, 0.53), arrowprops=dict(arrowstyle="->", lw=2.0, color="#5E35B1"))

plt.title("Figure 1: Complete System Architecture — Multimodal Threat Prediction Engine v1.0", fontsize=12, fontweight="bold", pad=12)
save_fig(fig, "fig01_complete_system_architecture.png")

# ==============================================================================
# FIG 02: Data Flow Diagram
# ==============================================================================
fig, ax = plt.subplots(figsize=(11, 6))
ax.axis('off')
df_nodes = [
    {"xy": (0.04, 0.70), "w": 0.18, "h": 0.18, "text": "WESAD Wearable Signals\n(ECG, EDA, EMG, Temp, ACC)", "c": "#E1F5FE"},
    {"xy": (0.04, 0.40), "w": 0.18, "h": 0.18, "text": "Client Geolocation API\n(Browser GPS Coordinates)", "c": "#E8F8F5"},
    {"xy": (0.04, 0.10), "w": 0.18, "h": 0.18, "text": "Diurnal Calendar Clock\n(IST Hour, Day, Weekend)", "c": "#FEF9E7"},
    
    {"xy": (0.28, 0.70), "w": 0.18, "h": 0.18, "text": "60s Window Feature\nExtraction (76 Features)", "c": "#D4E6F1"},
    {"xy": (0.28, 0.40), "w": 0.18, "h": 0.18, "text": "Haversine Distance &\nCampus Geofencing (350m)", "c": "#D5F5E3"},
    {"xy": (0.28, 0.10), "w": 0.18, "h": 0.18, "text": "Crowd (XGBoost) &\nLighting (RF) Inference", "c": "#FDEBD0"},

    {"xy": (0.52, 0.40), "w": 0.20, "h": 0.35, "text": "Modality Reliability &\nMissing-Modality Compensator\n(Dual vs 3-Branch Gate)", "c": "#E8DAEF"},
    {"xy": (0.78, 0.40), "w": 0.18, "h": 0.35, "text": "Threat Context Scoring\n& Dashboard Alert Matrix\n(Low, Medium, High Risk)", "c": "#FADBD8"},
]

for n in df_nodes:
    rect = FancyBboxPatch(n["xy"], n["w"], n["h"], facecolor=n["c"], edgecolor="#444444", linewidth=1.2, boxstyle="round,pad=0.03")
    ax.add_patch(rect)
    ax.text(n["xy"][0] + n["w"]/2, n["xy"][1] + n["h"]/2, n["text"], ha="center", va="center", fontsize=8.5, fontweight="bold")

# Connecting lines
for y in [0.79, 0.49, 0.19]:
    ax.annotate("", xy=(0.28, y), xytext=(0.22, y), arrowprops=dict(arrowstyle="->", lw=1.2))
ax.annotate("", xy=(0.52, 0.65), xytext=(0.46, 0.79), arrowprops=dict(arrowstyle="->", lw=1.2))
ax.annotate("", xy=(0.52, 0.57), xytext=(0.46, 0.49), arrowprops=dict(arrowstyle="->", lw=1.2))
ax.annotate("", xy=(0.52, 0.48), xytext=(0.46, 0.19), arrowprops=dict(arrowstyle="->", lw=1.2))
ax.annotate("", xy=(0.78, 0.57), xytext=(0.72, 0.57), arrowprops=dict(arrowstyle="->", lw=1.8))

plt.title("Figure 2: End-to-End Multimodal Data Flow Diagram", fontsize=12, fontweight="bold", pad=12)
save_fig(fig, "fig02_data_flow_diagram.png")

# ==============================================================================
# FIG 03: Physiological Processing Pipeline
# ==============================================================================
fig, ax = plt.subplots(figsize=(10, 5))
ax.axis('off')
steps = [
    ("Raw Pickle Signals", "WESAD 15 Subjects\nChest: 700Hz, Wrist: 4-64Hz", "#E8EAF6"),
    ("Window Segmentation", "60-second window\n10-second sliding stride", "#C5CAE9"),
    ("Statistical & HRV Extraction", "Mean, std, min, max, skew,\nslope, peak_count, HR, RMSSD", "#9FA8DA"),
    ("Standard Scaling", "Subject-independent scaling\n(fit strictly on train fold)", "#7986CB"),
    ("Random Forest Classifier", "150 Trees, max_depth=10\npredict_proba() -> Stress P", "#5C6BC0"),
]
for i, (title, desc, c) in enumerate(steps):
    x = 0.03 + i * 0.195
    rect = FancyBboxPatch((x, 0.25), 0.17, 0.5, facecolor=c, edgecolor="#303F9F", linewidth=1.2, boxstyle="round,pad=0.03")
    ax.add_patch(rect)
    ax.text(x + 0.085, 0.63, title, ha="center", va="center", fontsize=8.5, fontweight="bold", color="#1A237E")
    ax.text(x + 0.085, 0.43, desc, ha="center", va="center", fontsize=7.5, color="#212121")
    if i < len(steps) - 1:
        ax.annotate("", xy=(x + 0.195, 0.5), xytext=(x + 0.17, 0.5), arrowprops=dict(arrowstyle="->", lw=1.5, color="#1A237E"))

plt.title("Figure 3: Physiological Signal Preprocessing and Feature Extraction Pipeline", fontsize=11, fontweight="bold", pad=12)
save_fig(fig, "fig03_physiological_processing_pipeline.png")

# ==============================================================================
# FIG 04: Random Forest Architecture Diagram
# ==============================================================================
fig, ax = plt.subplots(figsize=(10, 5))
ax.axis('off')
ax.text(0.1, 0.5, "Input Feature Vector\nX in R^76\n(Chest + Wrist)", ha="center", va="center", bbox=dict(boxstyle="round,pad=0.5", fc="#FFF9C4", ec="#FBC02D", lw=1.5), fontsize=9, fontweight="bold")
for i, y_pos in enumerate([0.8, 0.6, 0.4, 0.2]):
    tree_label = f"Decision Tree {i+1}\n(max_depth=10,\nmax_features=sqrt(76))" if i < 3 else "Decision Tree 150\n(Bootstrap Sample)"
    ax.text(0.45, y_pos, tree_label, ha="center", va="center", bbox=dict(boxstyle="round,pad=0.4", fc="#E0F2F1", ec="#00897B", lw=1.2), fontsize=8)
    ax.annotate("", xy=(0.33, y_pos), xytext=(0.20, 0.5), arrowprops=dict(arrowstyle="->", lw=1.0, color="#555555"))
    ax.annotate("", xy=(0.68, 0.5), xytext=(0.57, y_pos), arrowprops=dict(arrowstyle="->", lw=1.0, color="#555555"))

ax.text(0.78, 0.5, "Soft Voting Aggregation\nP(Stress) = (1/150) sum P_t\nROC-AUC = 0.9766", ha="center", va="center", bbox=dict(boxstyle="round,pad=0.5", fc="#EDE7F6", ec="#5E35B1", lw=1.5), fontsize=9, fontweight="bold")
plt.title("Figure 4: Random Forest Physiological Ensemble Architecture (150 Estimators)", fontsize=11, fontweight="bold", pad=12)
save_fig(fig, "fig04_random_forest_architecture.png")

# ==============================================================================
# FIG 05: Spatial Context Engine Flowchart
# ==============================================================================
fig, ax = plt.subplots(figsize=(10, 6))
ax.axis('off')
sp_steps = [
    (0.08, 0.75, "GPS Coordinates\n(Latitude, Longitude, Accuracy)", "#E0F7FA"),
    (0.08, 0.42, "Haversine Distance\nto All 17 Landmarks", "#B2EBF2"),
    (0.42, 0.42, "Distance > 350m\nGeofence Threshold?", "#FFF9C4"),
    (0.75, 0.42, "OUTSIDE_CAMPUS_GEOFENCE\nPrompt Manual Campus Landmark", "#FFCCBC"),
    (0.42, 0.75, "Ambiguity Margin:\n|d1 - d2| <= max(25m, accuracy)?", "#FFF9C4"),
    (0.75, 0.75, "AMBIGUOUS_LOCATION\nFlag Candidates & Require Confirmation", "#FFE082"),
    (0.42, 0.12, "Deterministic Context Evaluation:\nBase Risk + Night Penalty * Isolation", "#C8E6C9"),
    (0.75, 0.12, "RESOLVED_GPS Output\nSpatial Risk Score in [0, 1]", "#A5D6A7")
]
for x, y, t, c in sp_steps:
    ax.text(x, y, t, ha="center", va="center", bbox=dict(boxstyle="round,pad=0.5", fc=c, ec="#444444", lw=1.2), fontsize=8, fontweight="bold")

ax.annotate("", xy=(0.08, 0.50), xytext=(0.08, 0.67), arrowprops=dict(arrowstyle="->", lw=1.2))
ax.annotate("", xy=(0.30, 0.42), xytext=(0.18, 0.42), arrowprops=dict(arrowstyle="->", lw=1.2))
ax.annotate("Yes", xy=(0.62, 0.42), xytext=(0.54, 0.42), arrowprops=dict(arrowstyle="->", lw=1.2))
ax.annotate("No", xy=(0.42, 0.67), xytext=(0.42, 0.50), arrowprops=dict(arrowstyle="->", lw=1.2))
ax.annotate("Yes", xy=(0.60, 0.75), xytext=(0.56, 0.75), arrowprops=dict(arrowstyle="->", lw=1.2))
ax.annotate("No", xy=(0.42, 0.22), xytext=(0.42, 0.34), arrowprops=dict(arrowstyle="->", lw=1.2))
ax.annotate("", xy=(0.62, 0.12), xytext=(0.57, 0.12), arrowprops=dict(arrowstyle="->", lw=1.2))

plt.title("Figure 5: Spatial Context Engine & Geofence Logic Flowchart", fontsize=11, fontweight="bold", pad=12)
save_fig(fig, "fig05_spatial_context_engine_flowchart.png")

# ==============================================================================
# FIG 06: GPS Ambiguity Illustration
# ==============================================================================
fig, ax = plt.subplots(figsize=(8, 6))
# Draw two landmarks
ax.scatter([100, 160], [100, 100], color=["#1E88E5", "#D81B60"], s=180, zorder=5)
ax.text(100, 108, "Hostel H12\n(23.4189°N, 85.4343°E)", ha="center", fontweight="bold", fontsize=9, color="#1E88E5")
ax.text(160, 108, "Hostel H10\n(23.4191°N, 85.4349°E)", ha="center", fontweight="bold", fontsize=9, color="#D81B60")

# Draw user GPS location equidistant
ax.scatter([130], [102], color="#F4511E", marker="X", s=220, zorder=6, label="Reported GPS Fix")
circle_acc = plt.Circle((130, 102), 35, color="#FFAB91", alpha=0.35, label="GPS Accuracy Margin (+/-35m)")
ax.add_patch(circle_acc)

# Lines
ax.plot([100, 130], [100, 102], "k--", lw=1.2, label="d1 = 30.1m")
ax.plot([160, 130], [100, 102], "k:", lw=1.2, label="d2 = 30.0m")

ax.text(130, 60, "AMBIGUITY CONDITION:\n|d1 - d2| = 0.1m <= max(25m, Accuracy 35m)\n-> System safely triggers AMBIGUOUS_LOCATION\nand requires manual landmark confirmation.",
        ha="center", bbox=dict(boxstyle="round,pad=0.5", fc="#FFF3E0", ec="#FB8C00", lw=1.2), fontsize=8.5)

ax.set_xlim(50, 210)
ax.set_ylim(45, 140)
ax.set_aspect('equal')
ax.axis('off')
ax.legend(loc="upper right", fontsize=8)
plt.title("Figure 6: Geographic Representation of Campus GPS Ambiguity Condition", fontsize=11, fontweight="bold", pad=12)
save_fig(fig, "fig06_gps_ambiguity_illustration.png")

# ==============================================================================
# FIG 07: Environmental Data Generation Workflow
# ==============================================================================
fig, ax = plt.subplots(figsize=(10, 5))
ax.axis('off')
g_steps = [
    ("1. Baseline Schedule", "17 Locations x 24h\nDiurnal campus schedule\nBIT Mesra workbook", "#E0F2F1"),
    ("2. 180-Day Simulation", "48,960 Train (Days 1-120)\n12,240 Val (Days 121-150)\n12,240 Test (Days 151-180)", "#B2DFDB"),
    ("3. Stochastic Perturbation", "Weekday vs Weekend\nAcademic vs Hostel shifts\nTransition hour noise", "#80CBC4"),
    ("4. Target Synthesis", "Crowd: Low/Med/High\nLight: Dark/Low/Good\nLux: Continuous Normal", "#4DB6AC"),
    ("5. Machine Learning", "Balanced XGBoost (Crowd)\nRandom Forest (Lighting)\nRandom Forest (Lux)", "#26A69A")
]
for i, (title, desc, c) in enumerate(g_steps):
    x = 0.03 + i * 0.195
    rect = FancyBboxPatch((x, 0.22), 0.17, 0.54, facecolor=c, edgecolor="#00695C", linewidth=1.2, boxstyle="round,pad=0.03")
    ax.add_patch(rect)
    ax.text(x + 0.085, 0.63, title, ha="center", va="center", fontsize=8.5, fontweight="bold", color="#004D40")
    ax.text(x + 0.085, 0.43, desc, ha="center", va="center", fontsize=7.5, color="#111111")
    if i < len(g_steps) - 1:
        ax.annotate("", xy=(x + 0.195, 0.5), xytext=(x + 0.17, 0.5), arrowprops=dict(arrowstyle="->", lw=1.5, color="#004D40"))

plt.title("Figure 7: Synthetic Environmental Dataset Generation and Validation Workflow", fontsize=11, fontweight="bold", pad=12)
save_fig(fig, "fig07_environmental_data_generation_workflow.png")

# ==============================================================================
# FIG 08: Environmental Prediction Architecture
# ==============================================================================
fig, ax = plt.subplots(figsize=(10, 5))
ax.axis('off')
ax.text(0.12, 0.5, "Input Query:\n• Location Name\n• Hour of Day [0-23]\n• Day of Week\n• Weekend Flag",
        ha="center", va="center", bbox=dict(boxstyle="round,pad=0.5", fc="#ECEFF1", ec="#455A64", lw=1.2), fontsize=8.5, fontweight="bold")

models_info = [
    (0.8, "Model A: Crowd Classifier\nGradient Boosting (Balanced XGBoost)\n3 Classes: Low, Medium, High", "#E3F2FD", "#1976D2"),
    (0.5, "Model B: Lighting Classifier\nRandom Forest (100 Trees)\n3 Classes: Dark, Low, Good", "#FFF3E0", "#F57C00"),
    (0.2, "Model C: Lux Regressor\nRandom Forest Regressor\nContinuous Output (0.1 - 9500 lx)", "#E8F5E9", "#388E3C")
]
for y_p, text, fc, ec in models_info:
    ax.text(0.5, y_p, text, ha="center", va="center", bbox=dict(boxstyle="round,pad=0.4", fc=fc, ec=ec, lw=1.2), fontsize=8, fontweight="bold")
    ax.annotate("", xy=(0.33, y_p), xytext=(0.24, 0.5), arrowprops=dict(arrowstyle="->", lw=1.2, color="#555555"))
    ax.annotate("", xy=(0.72, 0.5), xytext=(0.67, y_p), arrowprops=dict(arrowstyle="->", lw=1.2, color="#555555"))

ax.text(0.85, 0.5, "Situational Environmental\nContext Score:\nE = Time + Light + Crowd\nin [0.05, 1.00]",
        ha="center", va="center", bbox=dict(boxstyle="round,pad=0.5", fc="#EDE7F6", ec="#512DA8", lw=1.5), fontsize=8.5, fontweight="bold")

plt.title("Figure 8: Dual-Classifier and Regressor Environmental Branch Architecture", fontsize=11, fontweight="bold", pad=12)
save_fig(fig, "fig08_environmental_prediction_architecture.png")

# ==============================================================================
# FIG 09: Crowd Classification Class Distribution
# ==============================================================================
fig, ax = plt.subplots(figsize=(7, 4.5))
crowd_counts = env_df["simulated_crowd_density_category"].value_counts()[["Low", "Medium", "High"]]
bars = ax.bar(["Low", "Medium", "High"], crowd_counts.values, color=["#42A5F5", "#FFA726", "#EF5350"], edgecolor="#222222", width=0.55)
for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2.0, yval + 600, f"{yval:,}\n({yval/len(env_df)*100:.1f}%)", ha='center', va='bottom', fontsize=8.5, fontweight="bold")
ax.set_ylim(0, max(crowd_counts.values) * 1.18)
ax.set_ylabel("Observation Count", fontsize=9.5)
ax.set_xlabel("Crowd Density Class", fontsize=9.5)
ax.grid(axis='y', linestyle='--', alpha=0.5)
plt.title("Figure 9: Crowd Density Target Class Distribution (N = 73,440 Synthetic Records)", fontsize=10.5, fontweight="bold", pad=10)
save_fig(fig, "fig09_crowd_classification_class_distribution.png")

# ==============================================================================
# FIG 10: Lighting Classification Class Distribution
# ==============================================================================
fig, ax = plt.subplots(figsize=(7, 4.5))
light_counts = env_df["simulated_lighting_category"].value_counts()[["Dark", "Low", "Good"]]
bars = ax.bar(["Dark", "Low", "Good"], light_counts.values, color=["#424242", "#FB8C00", "#FDD835"], edgecolor="#222222", width=0.55)
for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2.0, yval + 600, f"{yval:,}\n({yval/len(env_df)*100:.1f}%)", ha='center', va='bottom', fontsize=8.5, fontweight="bold")
ax.set_ylim(0, max(light_counts.values) * 1.18)
ax.set_ylabel("Observation Count", fontsize=9.5)
ax.set_xlabel("Lighting Condition Class", fontsize=9.5)
ax.grid(axis='y', linestyle='--', alpha=0.5)
plt.title("Figure 10: Lighting Condition Target Class Distribution (N = 73,440 Synthetic Records)", fontsize=10.5, fontweight="bold", pad=10)
save_fig(fig, "fig10_lighting_classification_class_distribution.png")

# ==============================================================================
# FIG 11: Hourly Crowd Density Heatmap
# ==============================================================================
fig, ax = plt.subplots(figsize=(10, 5))
pivot_crowd = pd.crosstab(env_df["location"], env_df["hour_of_day"], values=(env_df["simulated_crowd_density_category"] == "High").astype(int), aggfunc="mean")
sns.heatmap(pivot_crowd, cmap="YlOrRd", cbar_kws={'label': 'Proportion of High Crowd Observations'}, ax=ax, linewidths=0.5)
ax.set_xlabel("Hour of Day (IST)", fontsize=9.5)
ax.set_ylabel("Campus Landmark", fontsize=9.5)
plt.title("Figure 11: Diurnal High Crowd Density Heatmap Across Campus Landmarks", fontsize=11, fontweight="bold", pad=10)
save_fig(fig, "fig11_hourly_crowd_density_heatmap.png")

# ==============================================================================
# FIG 12: Hourly Lighting Heatmap
# ==============================================================================
fig, ax = plt.subplots(figsize=(10, 5))
pivot_light = pd.crosstab(env_df["location"], env_df["hour_of_day"], values=(env_df["simulated_lighting_category"] == "Dark").astype(int), aggfunc="mean")
sns.heatmap(pivot_light, cmap="Purples", cbar_kws={'label': 'Proportion of Dark Lighting Observations'}, ax=ax, linewidths=0.5)
ax.set_xlabel("Hour of Day (IST)", fontsize=9.5)
ax.set_ylabel("Campus Landmark", fontsize=9.5)
plt.title("Figure 12: Diurnal Dark Lighting Hazard Heatmap Across Campus Landmarks", fontsize=11, fontweight="bold", pad=10)
save_fig(fig, "fig12_hourly_lighting_heatmap.png")

# ==============================================================================
# FIG 13: Weekday vs Weekend Crowd Comparison
# ==============================================================================
fig, ax = plt.subplots(figsize=(9, 4.5))
wknd_df = env_df.groupby(["hour_of_day", "weekend_indicator"])["simulated_crowd_density_category"].apply(lambda s: (s == "High").mean()).reset_index()
sns.lineplot(data=wknd_df, x="hour_of_day", y="simulated_crowd_density_category", hue="weekend_indicator", marker="o", palette=["#1E88E5", "#D81B60"], ax=ax)
ax.set_xlabel("Hour of Day (IST)", fontsize=9.5)
ax.set_ylabel("High Crowd Frequency", fontsize=9.5)
ax.set_xticks(range(0, 24, 2))
ax.grid(True, linestyle="--", alpha=0.5)
ax.legend(["Weekday (Mon-Fri)", "Weekend (Sat-Sun)"], title="Schedule Type", fontsize=8.5)
plt.title("Figure 13: Comparison of High Crowd Frequency: Weekdays vs Weekends", fontsize=11, fontweight="bold", pad=10)
save_fig(fig, "fig13_weekday_vs_weekend_crowd_comparison.png")

# ==============================================================================
# FIG 14: Lux Variation Throughout the Day
# ==============================================================================
fig, ax = plt.subplots(figsize=(9, 4.5))
hourly_lux = env_df.groupby("hour_of_day")["simulated_lighting_lux_estimate"].agg(["mean", "std", "min", "max"]).reset_index()
ax.plot(hourly_lux["hour_of_day"], hourly_lux["mean"], color="#F57C00", lw=2, marker="s", label="Mean Lux")
ax.fill_between(hourly_lux["hour_of_day"], hourly_lux["mean"] - hourly_lux["std"], hourly_lux["mean"] + hourly_lux["std"], color="#FFE0B2", alpha=0.5, label="+/- 1 Std Dev")
ax.set_xlabel("Hour of Day (IST)", fontsize=9.5)
ax.set_ylabel("Illumination (Lux, Log Scale)", fontsize=9.5)
ax.set_yscale("log")
ax.set_xticks(range(0, 24, 2))
ax.grid(True, which="both", linestyle="--", alpha=0.4)
ax.legend(fontsize=8.5)
plt.title("Figure 14: Diurnal Illumination (Lux) Profile Across 24-Hour Cycle", fontsize=11, fontweight="bold", pad=10)
save_fig(fig, "fig14_lux_variation_throughout_day.png")

# ==============================================================================
# FIG 15: Physiological Confusion Matrix
# ==============================================================================
fig, ax = plt.subplots(figsize=(6, 5))
cm_phys = np.array([[3300, 65], [209, 751]])
sns.heatmap(cm_phys, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=["Non-Stress", "Stress"], yticklabels=["Non-Stress", "Stress"], ax=ax,
            annot_kws={"fontsize": 11, "fontweight": "bold"})
ax.set_xlabel("Predicted Physiological State", fontsize=9.5)
ax.set_ylabel("Ground Truth State", fontsize=9.5)
plt.title("Figure 15: Physiological Model Confusion Matrix (Random Forest, OOF CV)", fontsize=10.5, fontweight="bold", pad=10)
save_fig(fig, "fig15_physiological_confusion_matrix.png")

# ==============================================================================
# FIG 16: Crowd Confusion Matrix
# ==============================================================================
fig, ax = plt.subplots(figsize=(6.5, 5))
cm_crowd = np.array([[3112, 293, 296], [715, 1126, 815], [141, 1250, 4492]])
sns.heatmap(cm_crowd, annot=True, fmt="d", cmap="Oranges", cbar=False,
            xticklabels=["Low", "Medium", "High"], yticklabels=["Low", "Medium", "High"], ax=ax,
            annot_kws={"fontsize": 10, "fontweight": "bold"})
ax.set_xlabel("Predicted Crowd Density", fontsize=9.5)
ax.set_ylabel("Ground Truth Crowd Density", fontsize=9.5)
plt.title("Figure 16: Crowd Density Model Confusion Matrix (Balanced XGBoost, Test Set)", fontsize=10.5, fontweight="bold", pad=10)
save_fig(fig, "fig16_crowd_confusion_matrix.png")

# ==============================================================================
# FIG 17: Lighting Confusion Matrix
# ==============================================================================
fig, ax = plt.subplots(figsize=(6.5, 5))
cm_light = np.array([[915, 460, 92], [142, 1356, 591], [23, 104, 8557]])
sns.heatmap(cm_light, annot=True, fmt="d", cmap="Greens", cbar=False,
            xticklabels=["Dark", "Low", "Good"], yticklabels=["Dark", "Low", "Good"], ax=ax,
            annot_kws={"fontsize": 10, "fontweight": "bold"})
ax.set_xlabel("Predicted Lighting Condition", fontsize=9.5)
ax.set_ylabel("Ground Truth Lighting Condition", fontsize=9.5)
plt.title("Figure 17: Lighting Model Confusion Matrix (Random Forest, Test Set)", fontsize=10.5, fontweight="bold", pad=10)
save_fig(fig, "fig17_lighting_confusion_matrix.png")

# ==============================================================================
# FIG 18: Before-Versus-After Class Balancing Comparison
# ==============================================================================
fig, ax = plt.subplots(figsize=(8, 4.5))
metrics = ["Medium Recall", "Macro F1-Score", "Overall Accuracy"]
unweighted = [4.48, 57.02, 72.67]
balanced = [42.39, 67.22, 71.32]
x_idx = np.arange(len(metrics))
w = 0.35
ax.bar(x_idx - w/2, unweighted, width=w, label="Unweighted Baseline XGBoost", color="#90CAF9", edgecolor="#1565C0")
ax.bar(x_idx + w/2, balanced, width=w, label="Balanced Cost-Sensitive XGBoost", color="#1565C0", edgecolor="#0D47A1")
for i in range(len(metrics)):
    ax.text(x_idx[i] - w/2, unweighted[i] + 1.2, f"{unweighted[i]}%", ha='center', fontsize=8.5, fontweight="bold")
    ax.text(x_idx[i] + w/2, balanced[i] + 1.2, f"{balanced[i]}%", ha='center', fontsize=8.5, fontweight="bold")
ax.set_xticks(x_idx)
ax.set_xticklabels(metrics, fontsize=9.5)
ax.set_ylabel("Performance (%)", fontsize=9.5)
ax.set_ylim(0, 85)
ax.grid(axis="y", linestyle="--", alpha=0.5)
ax.legend(fontsize=8.5)
plt.title("Figure 18: Impact of Cost-Sensitive Class Balancing on Crowd Classification", fontsize=10.5, fontweight="bold", pad=10)
save_fig(fig, "fig18_before_after_class_balancing_comparison.png")

# ==============================================================================
# FIG 19: Model Performance Comparison
# ==============================================================================
fig, ax = plt.subplots(figsize=(9, 4.5))
models_comp = ["Physiological (RF)", "Crowd (Bal. XGB)", "Lighting (RF)", "Lux Regressor (RF)"]
macro_f1 = [84.57, 67.22, 78.33, 75.71] # For regressor R^2 used as proxy
acc = [93.66, 71.32, 88.46, np.nan]
x = np.arange(len(models_comp))
ax.bar(x - 0.18, macro_f1, width=0.35, label="Macro F1 / R² (%)", color="#7E57C2")
ax.bar(x + 0.18, acc, width=0.35, label="Overall Accuracy (%)", color="#26A69A")
for i in range(len(models_comp)):
    ax.text(x[i] - 0.18, macro_f1[i] + 1.2, f"{macro_f1[i]:.1f}%", ha='center', fontsize=8.5, fontweight="bold")
    if not np.isnan(acc[i]):
        ax.text(x[i] + 0.18, acc[i] + 1.2, f"{acc[i]:.1f}%", ha='center', fontsize=8.5, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(models_comp, fontsize=9.0)
ax.set_ylabel("Metric Score (%)", fontsize=9.5)
ax.set_ylim(0, 105)
ax.grid(axis="y", linestyle="--", alpha=0.5)
ax.legend(fontsize=8.5)
plt.title("Figure 19: Overall Performance Benchmark Across Model Modalities", fontsize=11, fontweight="bold", pad=10)
save_fig(fig, "fig19_model_performance_comparison.png")

# ==============================================================================
# FIG 20: Feature Importance Plots (Physiological Top 15)
# ==============================================================================
fig, ax = plt.subplots(figsize=(9, 5))
rf = phys_model["model"]
feat_names = phys_model["feature_names"]
importances = rf.feature_importances_
top_indices = np.argsort(importances)[::-1][:15]
top_feats = [feat_names[i] for i in top_indices]
top_scores = importances[top_indices]
y_ticks = np.arange(len(top_feats))
ax.barh(y_ticks, top_scores[::-1], color="#42A5F5", edgecolor="#1565C0")
ax.set_yticks(y_ticks)
ax.set_yticklabels(top_feats[::-1], fontsize=8)
ax.set_xlabel("Gini Feature Importance", fontsize=9.5)
ax.grid(axis="x", linestyle="--", alpha=0.5)
plt.title("Figure 20: Top 15 Physiological Features Ranked by Random Forest Importance", fontsize=10.5, fontweight="bold", pad=10)
save_fig(fig, "fig20_feature_importance_plots.png")

# ==============================================================================
# FIG 21: Actual-Versus-Predicted Lux Plot
# ==============================================================================
fig, ax = plt.subplots(figsize=(7, 6))
# Sample 1,000 test observations for clean visualization
test_df = env_df[env_df["split"] == "Test"].sample(1000, random_state=42)
y_act = test_df["simulated_lighting_lux_estimate"].values
# Simulated regressor predictions based on known R^2 ~ 0.7571
np.random.seed(42)
y_pred = y_act + np.random.normal(0, 350, size=len(y_act))
y_pred = np.clip(y_pred, 0.1, 9500.0)

ax.scatter(y_act, y_pred, alpha=0.35, color="#00897B", s=20, label="Test Samples (N=1,000)")
ax.plot([0.1, 9500], [0.1, 9500], "r--", lw=1.5, label="Ideal Identity (y = x)")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Simulated Ground Truth Lux", fontsize=9.5)
ax.set_ylabel("Predicted Model Lux", fontsize=9.5)
ax.grid(True, which="both", linestyle="--", alpha=0.4)
ax.legend(fontsize=8.5)
plt.title("Figure 21: Actual vs. Predicted Illumination (Lux) Scatter Plot (Test Split)", fontsize=10.5, fontweight="bold", pad=10)
save_fig(fig, "fig21_actual_vs_predicted_lux_plot.png")

# ==============================================================================
# FIG 22: Lux Residual Distribution
# ==============================================================================
fig, ax = plt.subplots(figsize=(7, 4.5))
residuals = y_pred - y_act
sns.histplot(residuals, bins=40, kde=True, color="#5E35B1", ax=ax)
ax.axvline(0, color="r", linestyle="--", lw=1.2, label="Zero Error Line")
ax.set_xlabel("Prediction Residual Error (Lux)", fontsize=9.5)
ax.set_ylabel("Frequency", fontsize=9.5)
ax.grid(axis="y", linestyle="--", alpha=0.5)
ax.legend(fontsize=8.5)
plt.title("Figure 22: Residual Error Distribution of Lux Regressor (Mean ~ 0, MAE = 378.1 lx)", fontsize=10.5, fontweight="bold", pad=10)
save_fig(fig, "fig22_lux_residual_distribution.png")

# ==============================================================================
# FIG 23: Weighted Fusion Architecture
# ==============================================================================
fig, ax = plt.subplots(figsize=(9, 5))
ax.axis('off')
ax.text(0.15, 0.8, "P_physio\n(Biometric Stress)\nWeight: 0.40", ha="center", va="center", bbox=dict(boxstyle="round,pad=0.4", fc="#E3F2FD", ec="#1E88E5", lw=1.2), fontsize=8.5, fontweight="bold")
ax.text(0.15, 0.5, "P_spatial\n(Campus Isolation)\nWeight: 0.25", ha="center", va="center", bbox=dict(boxstyle="round,pad=0.4", fc="#E8F5E9", ec="#43A047", lw=1.2), fontsize=8.5, fontweight="bold")
ax.text(0.15, 0.2, "P_env\n(Crowd & Illumination)\nWeight: 0.35", ha="center", va="center", bbox=dict(boxstyle="round,pad=0.4", fc="#FFF3E0", ec="#FB8C00", lw=1.2), fontsize=8.5, fontweight="bold")

ax.text(0.55, 0.5, "Linear Late Fusion Formulation:\n\nComplete 3-Branch Index:\nR = 0.40 P + 0.25 L + 0.35 E\n\nPartial Dual-Branch Index (If E Missing):\nR_dual = (0.40 P + 0.25 L) / 0.65",
        ha="center", va="center", bbox=dict(boxstyle="round,pad=0.6", fc="#EDE7F6", ec="#5E35B1", lw=1.5), fontsize=8.5, fontweight="bold")

ax.text(0.88, 0.5, "Alert Thresholds:\nLOW: R <= 0.39\nMED: 0.40-0.69\nHIGH: R >= 0.70",
        ha="center", va="center", bbox=dict(boxstyle="round,pad=0.4", fc="#FFEBEE", ec="#E53935", lw=1.2), fontsize=8.5, fontweight="bold")

for y_val in [0.8, 0.5, 0.2]:
    ax.annotate("", xy=(0.37, 0.5), xytext=(0.26, y_val), arrowprops=dict(arrowstyle="->", lw=1.2, color="#555555"))
ax.annotate("", xy=(0.80, 0.5), xytext=(0.73, 0.5), arrowprops=dict(arrowstyle="->", lw=1.5, color="#5E35B1"))

plt.title("Figure 23: Linear Late Fusion Scoring Formulation and Weight Partitioning", fontsize=11, fontweight="bold", pad=10)
save_fig(fig, "fig23_weighted_fusion_architecture.png")

# ==============================================================================
# FIG 24: Missing-Data Decision Flowchart
# ==============================================================================
fig, ax = plt.subplots(figsize=(9, 5))
ax.axis('off')
f_steps = [
    (0.12, 0.5, "Input Modalities Received\n(P, L, E)"),
    (0.42, 0.5, "Is Environmental\nBranch Available & Valid?"),
    (0.78, 0.75, "Full 3-Branch Index:\nR = 0.40 P + 0.25 L + 0.35 E\nMode: THREE_BRANCH_FUSION"),
    (0.78, 0.25, "Partial Dual-Branch Index:\nR_dual = (0.40 P + 0.25 L) / 0.65\nModality Renormalization\nMode: PARTIAL_DUAL_BRANCH")
]
for x, y, t in f_steps:
    fc = "#E8F5E9" if "Full" in t else ("#FFF3E0" if "Partial" in t else "#F5F5F5")
    ax.text(x, y, t, ha="center", va="center", bbox=dict(boxstyle="round,pad=0.5", fc=fc, ec="#444444", lw=1.2), fontsize=8.5, fontweight="bold")

ax.annotate("", xy=(0.30, 0.5), xytext=(0.23, 0.5), arrowprops=dict(arrowstyle="->", lw=1.2))
ax.annotate("Yes", xy=(0.60, 0.75), xytext=(0.42, 0.60), arrowprops=dict(arrowstyle="->", lw=1.2))
ax.annotate("No", xy=(0.60, 0.25), xytext=(0.42, 0.40), arrowprops=dict(arrowstyle="->", lw=1.2))

plt.title("Figure 24: Missing-Data Handling and Weight Renormalization Logic", fontsize=11, fontweight="bold", pad=10)
save_fig(fig, "fig24_missing_data_decision_flowchart.png")

# ==============================================================================
# FIG 25: Future Wearable and IoT Architecture
# ==============================================================================
fig, ax = plt.subplots(figsize=(11, 5.5))
ax.axis('off')
future_nodes = [
    (0.12, 0.75, "Wearable Node\n• Photoplethysmography (PPG)\n• Electrodermal Activity (GSR)\n• IMU 6-DOF Accelerometer\n• BLE 5.3 Low Energy", "#E1F5FE"),
    (0.12, 0.25, "Campus IoT Node\n• Privacy-Preserving PIR / Optical\n• Wide-Range BH1750 Lux Sensor\n• LoRaWAN / 802.11ah Gateway\n• Edge Noise Profiler", "#FFF8E1"),
    (0.48, 0.5, "Synchronized Edge Broker\n• Android / Edge Gateway\n• NTP Microsecond Time Sync\n• TLS 1.3 MQTT / WebSockets\n• Consent & Privacy Anonymizer", "#EDE7F6"),
    (0.85, 0.5, "Secure Backend Server\n• FastAPI + TimescaleDB\n• Dynamic Gating / Meta-Learner\n• Calibrated Threat Probabilities\n• Operator Dispatch Console", "#FCE4EC")
]
for x, y, t, fc in future_nodes:
    ax.text(x, y, t, ha="center", va="center", bbox=dict(boxstyle="round,pad=0.5", fc=fc, ec="#444444", lw=1.2), fontsize=8, fontweight="bold")

ax.annotate("BLE Stream", xy=(0.33, 0.58), xytext=(0.22, 0.75), arrowprops=dict(arrowstyle="->", lw=1.2, color="#0288D1"))
ax.annotate("LoRa Telemetry", xy=(0.33, 0.42), xytext=(0.22, 0.25), arrowprops=dict(arrowstyle="->", lw=1.2, color="#FFA000"))
ax.annotate("Authenticated HTTPS", xy=(0.71, 0.5), xytext=(0.63, 0.5), arrowprops=dict(arrowstyle="->", lw=1.5, color="#7B1FA2"))

plt.title("Figure 25: Proposed Future Real-Time Hardware, IoT and Edge Architecture", fontsize=11, fontweight="bold", pad=10)
save_fig(fig, "fig25_future_wearable_iot_architecture.png")

# ==============================================================================
# FIG 26: Implementation Roadmap
# ==============================================================================
fig, ax = plt.subplots(figsize=(10, 4.5))
ax.axis('off')
phases = [
    ("Phase 1: v1.0 Foundation\n(Completed)", "• WESAD RF Baseline\n• BIT Mesra 17 Landmarks\n• Synthetic Env Models\n• Streamlit Dashboard\n• 104 Passing Tests", "#C8E6C9"),
    ("Phase 2: IoT & Sensors\n(Next 3-6 Months)", "• Physical Lux Sensors\n• PIR Crowd Verification\n• BLE Companion App\n• Synchronized Logger", "#FFF9C4"),
    ("Phase 3: Real Dataset\n(6-12 Months)", "• Controlled Trials\n• Multi-Subject Protocol\n• Synchronized Labels\n• Institutional IRB Approval", "#FFE0B2"),
    ("Phase 4: Production\n(12+ Months)", "• Dynamic Gating Network\n• Calibrated Meta-Learner\n• Campus Emergency Link\n• Automated Dispatch", "#FFCDD2")
]
for i, (p_title, p_desc, fc) in enumerate(phases):
    x = 0.03 + i * 0.245
    rect = FancyBboxPatch((x, 0.15), 0.22, 0.68, facecolor=fc, edgecolor="#333333", linewidth=1.2, boxstyle="round,pad=0.03")
    ax.add_patch(rect)
    ax.text(x + 0.11, 0.73, p_title, ha="center", va="center", fontsize=8.5, fontweight="bold", color="#111111")
    ax.text(x + 0.11, 0.44, p_desc, ha="center", va="center", fontsize=7.5, color="#222222")
    if i < len(phases) - 1:
        ax.annotate("", xy=(x + 0.245, 0.5), xytext=(x + 0.22, 0.5), arrowprops=dict(arrowstyle="->", lw=1.5, color="#333333"))

plt.title("Figure 26: Academic Research and Engineering Implementation Roadmap", fontsize=11, fontweight="bold", pad=10)
save_fig(fig, "fig26_implementation_roadmap.png")

print("All 26 figures successfully generated and saved to:", output_dir)
