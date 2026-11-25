# Multimodal Threat Prediction Engine — Version 1.0
## Technical Documentation: Academic Figure and Table Index for 30-Page B.Tech Project Report

---

### 1. Master Figure Index (All 26 Generated Publication Figures)

All 26 figures have been generated at **300 DPI publication quality** and are archived in the `REPORT_GENERATION_PACKAGE/FIGURES/` directory.

| Figure Filename | Suggested Academic Caption | Data Source & Provenance | Nature of Figure | Suggested Report Chapter |
| :--- | :--- | :--- | :---: | :---: |
| `fig01_complete_system_architecture.png` | **Figure 1.1:** Complete multi-branch system architecture of the Multimodal Threat Prediction Engine (v1.0). | System design specification (`app/main.py`, `src/fusion/`) | Conceptual / Architectural | **Chapter 1 / 3** (Introduction & Architecture) |
| `fig02_data_flow_diagram.png` | **Figure 3.1:** End-to-end data flow pipeline from raw sensing to dashboard alert generation. | System data contracts (`src/`) | Conceptual / Process | **Chapter 3** (System Design & Methodology) |
| `fig03_physiological_processing_pipeline.png`| **Figure 4.1:** Physiological signal preprocessing, 60s sliding window segmentation, and HRV extraction. | `src/physiological/feature_extraction.py` | Conceptual / Algorithmic | **Chapter 4** (Physiological Distress Branch) |
| `fig04_random_forest_architecture.png` | **Figure 4.2:** Random Forest ensemble architecture (150 decision trees) for biometric stress classification. | `src/physiological/train.py`, `physiological_model.pkl` | Conceptual / Model | **Chapter 4** (Physiological Distress Branch) |
| `fig05_spatial_context_engine_flowchart.png` | **Figure 5.1:** Spatial context engine decision flowchart illustrating geofencing and ambiguity gates. | `src/location/campus_map.py`, `src/environment/model.py`| Algorithmic Flowchart | **Chapter 5** (Spatial Context & Campus GIS) |
| `fig06_gps_ambiguity_illustration.png` | **Figure 5.2:** Geometric representation of GPS ambiguity condition between adjacent hostel landmarks. | `src/location/landmarks.py`, Surveyed Coordinates | Geometric / Conceptual | **Chapter 5** (Spatial Context & Campus GIS) |
| `fig07_environmental_data_generation_workflow.png`| **Figure 6.1:** Synthetic environmental dataset generation, schedule simulation, and chronological splitting. | `src/environment/generate_synthetic_dataset.py` | Synthetic Simulation Flow | **Chapter 6** (Environmental AI Branch) |
| `fig08_environmental_prediction_architecture.png`| **Figure 6.2:** Multi-output environmental architecture combining crowd, lighting, and lux inference. | `src/environment/model.py` | Architectural Flow | **Chapter 6** (Environmental AI Branch) |
| `fig09_crowd_classification_class_distribution.png`| **Figure 6.3:** Target class distribution for simulated crowd density ($N = 73,440$). | `BIT_Mesra_Synthetic_Environmental_Dataset.csv` | Synthetic Empirical Data | **Chapter 6** (Environmental AI Branch) |
| `fig10_lighting_classification_class_distribution.png`| **Figure 6.4:** Target class distribution for simulated lighting condition ($N = 73,440$). | `BIT_Mesra_Synthetic_Environmental_Dataset.csv` | Synthetic Empirical Data | **Chapter 6** (Environmental AI Branch) |
| `fig11_hourly_crowd_density_heatmap.png` | **Figure 6.5:** Diurnal high crowd density frequency heatmap across all 17 campus landmarks. | `BIT_Mesra_Synthetic_Environmental_Dataset.csv` | Synthetic Schedule Data | **Chapter 6** (Environmental AI Branch) |
| `fig12_hourly_lighting_heatmap.png` | **Figure 6.6:** Diurnal dark lighting hazard frequency heatmap across all 17 campus landmarks. | `BIT_Mesra_Synthetic_Environmental_Dataset.csv` | Synthetic Schedule Data | **Chapter 6** (Environmental AI Branch) |
| `fig13_weekday_vs_weekend_crowd_comparison.png`| **Figure 6.7:** Comparative line plot of high crowd density frequency: Weekdays vs. Weekends. | `BIT_Mesra_Synthetic_Environmental_Dataset.csv` | Synthetic Schedule Data | **Chapter 6** (Environmental AI Branch) |
| `fig14_lux_variation_throughout_day.png` | **Figure 6.8:** Diurnal continuous illuminance (lux) profile across 24 hours on a logarithmic scale. | `BIT_Mesra_Synthetic_Environmental_Dataset.csv` | Synthetic Simulation Data | **Chapter 6** (Environmental AI Branch) |
| `fig15_physiological_confusion_matrix.png` | **Figure 7.1:** Out-of-fold confusion matrix for the physiological Random Forest model ($N = 4,325$). | WESAD GroupKFold OOF Validation (`physiological_model.pkl`)| **Empirical Experimental Data** | **Chapter 7** (Results & Model Evaluation) |
| `fig16_crowd_confusion_matrix.png` | **Figure 7.2:** Confusion matrix of the Balanced XGBoost crowd model on chronological test split ($N = 12,240$). | Environmental Test Split (`crowd_model.pkl`) | Synthetic Experimental Data | **Chapter 7** (Results & Model Evaluation) |
| `fig17_lighting_confusion_matrix.png` | **Figure 7.3:** Confusion matrix of the Random Forest lighting model on chronological test split ($N = 12,240$).| Environmental Test Split (`lighting_model.pkl`)| Synthetic Experimental Data | **Chapter 7** (Results & Model Evaluation) |
| `fig18_before_after_class_balancing_comparison.png`| **Figure 7.4:** Impact of cost-sensitive class balancing on minority Medium crowd recall and Macro F1. | Comparative Model Evaluation (`crowd_model_comparison.csv`)| Comparative Benchmark | **Chapter 7** (Results & Model Evaluation) |
| `fig19_model_performance_comparison.png` | **Figure 7.5:** Comprehensive benchmark comparison of accuracy and Macro F1 across model branches. | Master Experimental Evaluation Tables | Comparative Benchmark | **Chapter 7** (Results & Model Evaluation) |
| `fig20_feature_importance_plots.png` | **Figure 7.6:** Top 15 physiological features ranked by Random Forest Gini impurity reduction. | Trained Model Artifact (`physiological_model.pkl`) | **Empirical Experimental Data** | **Chapter 7** (Results & Model Evaluation) |
| `fig21_actual_vs_predicted_lux_plot.png` | **Figure 7.7:** Actual vs. predicted continuous illuminance scatter plot for the Random Forest lux regressor. | Test Split Evaluation ($R^2 = 0.7571$) | Synthetic Experimental Data | **Chapter 7** (Results & Model Evaluation) |
| `fig22_lux_residual_distribution.png` | **Figure 7.8:** Residual prediction error distribution of the lux regressor ($\text{MAE} = 378.1\,\text{lx}$). | Test Split Residual Analysis | Synthetic Experimental Data | **Chapter 7** (Results & Model Evaluation) |
| `fig23_weighted_fusion_architecture.png` | **Figure 8.1:** Linear late-fusion scoring formulation, weight allocations, and alert threshold boundaries. | `src/fusion/predict.py`, `config.py` | Mathematical / Conceptual | **Chapter 8** (Multimodal Threat Fusion) |
| `fig24_missing_data_decision_flowchart.png` | **Figure 8.2:** Missing-data decision logic and weight renormalization flow for partial dual-branch fusion. | `src/fusion/fusion_engine.py` | Algorithmic Flowchart | **Chapter 8** (Multimodal Threat Fusion) |
| `fig25_future_wearable_iot_architecture.png` | **Figure 9.1:** Proposed four-tier future Internet-of-Things (IoT) hardware, BLE, and cloud architecture. | System Proposal (`11_FUTURE_SCOPE_AND_LIMITATIONS.md`) | Proposed Future Architecture | **Chapter 9** (Future Work & Hardware) |
| `fig26_implementation_roadmap.png` | **Figure 9.2:** Multi-phase engineering and academic research roadmap toward full campus deployment. | Project Roadmap (`11_FUTURE_SCOPE_AND_LIMITATIONS.md`) | Strategic Roadmap | **Chapter 9** (Future Work & Conclusion) |

---

### 2. Recommended Academic Report Tables

The table below outlines the core academic tables recommended for the 30-page project report, identifying their exact data sources in the package:

| Table Number | Recommended Academic Caption | Primary Data Source File | Location in Package | Suggested Report Chapter |
| :---: | :--- | :--- | :--- | :---: |
| **Table 2.1** | Literature Review Summary of Wearable Stress & Campus Safety Systems | `13_RESEARCH_REFERENCES.md` | Root Documentation | **Chapter 2** (Literature Review) |
| **Table 3.1** | Component Dependency Matrix and Module Data Flow | `02_REPOSITORY_ARCHITECTURE.md` | Root Documentation | **Chapter 3** (System Architecture) |
| **Table 4.1** | Synchronized Sensor Modalities and Sampling Rates in WESAD | `03_WESAD_DATASET_AND_PHYSIOLOGICAL_PIPELINE.md` | Root Documentation | **Chapter 4** (Physiological Branch) |
| **Table 4.2** | Complete 76-Dimensional Physiological Feature Dictionary | `physiological_feature_dictionary.csv` | `DATASET_SUMMARIES/` | **Chapter 4** (Physiological Branch) |
| **Table 4.3** | Subject-Wise Sample Distribution Across WESAD Experimental Conditions | `wesad_subject_audit.json`, `wesad_summary.json` | `EXPERIMENTAL_EVIDENCE/`| **Chapter 4** (Physiological Branch) |
| **Table 5.1** | BIT Mesra Campus Landmarks: Surveyed Coordinates and Vulnerability Indices | `campus_landmarks.csv` | `DATASET_SUMMARIES/` | **Chapter 5** (Spatial Context Branch) |
| **Table 5.2** | Standardized Geospatial Engine Status Codes and Decision Rules | `04_SPATIAL_CONTEXT_ENGINE.md` | Root Documentation | **Chapter 5** (Spatial Context Branch) |
| **Table 6.1** | Baseline Diurnal Operating Schedule Rules for 17 Campus Landmarks | `05_ENVIRONMENTAL_DATASET_GENERATION.md` | Root Documentation | **Chapter 6** (Environmental AI Branch) |
| **Table 6.2** | Synthetic Environmental Dataset Statistical Summary ($N = 73,440$) | `environmental_summary_statistics.csv` | `DATASET_SUMMARIES/` | **Chapter 6** (Environmental AI Branch) |
| **Table 7.1** | Multi-Algorithm Performance Comparison on Empirical WESAD Dataset | `08_COMPLETE_EXPERIMENTAL_RESULTS.md` | Root Documentation | **Chapter 7** (Experimental Results) |
| **Table 7.2** | Crowd Density Classifier Comparison on Held-Out Test Set | `crowd_model_comparison.csv` | `EXPERIMENTAL_EVIDENCE/`| **Chapter 7** (Experimental Results) |
| **Table 7.3** | Lighting Condition Classifier Comparison on Held-Out Test Set | `lighting_model_comparison.csv` | `EXPERIMENTAL_EVIDENCE/`| **Chapter 7** (Experimental Results) |
| **Table 7.4** | Multi-Branch Stacking Simulation Benchmark Comparison | `stacking_benchmark_results_synthetic_demo.csv` | `EXPERIMENTAL_EVIDENCE/`| **Chapter 7** (Experimental Results) |
| **Table 8.1** | Five Worked Numerical Threat Evaluation Scenarios Under Baseline Weights | `07_MULTIMODAL_FUSION_AND_SCORING.md` | Root Documentation | **Chapter 8** (Multimodal Fusion) |
| **Table 9.1** | Automated Software Test Suite Verification Matrix (104 Tests, 18 Files) | `10_SOFTWARE_TESTING_AND_VERIFICATION.md` | Root Documentation | **Chapter 9** (Software Verification) |
| **Table 10.1**| Production Technology Stack, Installed Package Versions, and Environment | `12_TECHNOLOGY_STACK_AND_ENVIRONMENT.md` | Root Documentation | **Chapter 10** (Implementation Specs) |
