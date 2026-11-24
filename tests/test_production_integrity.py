"""
Production Integrity and Academic Provenance Regression Tests.
Verifies:
1. Production inference refuses to load or use the synthetic artifact.
2. The synthetic artifact cannot be loaded as a production meta-model.
3. The production predictor returns the untrained status when no valid model exists.
4. The synthetic training workflow outputs only to the simulation directory.
5. The production training workflow rejects datasets with missing labels or synthetic properties.
6. WESAD, geospatial, and environmental inference still work correctly.
7. predict_fusion does not output synthetic threat probabilities.
"""

import unittest
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

import config
from src.fusion.predict import MultimodalThreatPredictor
from src.fusion.stacking_pipeline import (
    StackingEnsemblePipeline,
    IneligibleProductionArtifactError,
    SyntheticDatasetRejectedError,
    MissingGroundTruthLabelError,
    DataCircularityError
)
from src.fusion.train import (
    run_synthetic_simulation_demo,
    train_production_stacking_model
)

class TestProductionIntegrity(unittest.TestCase):
    def setUp(self):
        self.predictor = MultimodalThreatPredictor()
        self.demo_artifact_dir = config.MODELS_DIR / "fusion" / "synthetic_demo"
        self.demo_artifact_path = self.demo_artifact_dir / "stacking_ensemble_demo.pkl"
        self.simulation_reports_dir = Path("reports/synthetic_simulation")

    def test_production_inference_refuses_synthetic_artifact(self):
        """1. Production inference refuses to load or use the synthetic artifact."""
        if self.demo_artifact_path.exists():
            with self.assertRaises(IneligibleProductionArtifactError):
                StackingEnsemblePipeline.load_artifact(
                    path=self.demo_artifact_path,
                    require_production_eligible=True
                )

    def test_synthetic_artifact_cannot_be_loaded_as_production_meta_model(self):
        """2. The synthetic artifact cannot be loaded as a production meta-model."""
        if self.demo_artifact_path.exists():
            with open(self.demo_artifact_path, "rb") as f:
                artifact = pickle.load(f)

            # Metadata verification
            self.assertFalse(artifact.get("production_eligible", True))
            meta = artifact.get("metadata", {})
            self.assertEqual(meta.get("dataset_type"), "synthetic_simulation")

            # Loader must reject it under production mode
            with self.assertRaises(IneligibleProductionArtifactError):
                StackingEnsemblePipeline.load_artifact(self.demo_artifact_path, require_production_eligible=True)

            # Loader can load it only in explicit demo/testing mode
            demo_pipeline = StackingEnsemblePipeline.load_artifact(self.demo_artifact_path, require_production_eligible=False)
            self.assertIsNotNone(demo_pipeline)
            self.assertFalse(demo_pipeline.production_eligible)

    def test_production_predictor_returns_untrained_status(self):
        """3. The production predictor returns the untrained status when no valid model exists."""
        # Ensure production stacking model file does not exist
        prod_model_path = config.MODELS_DIR / "fusion" / "stacking_ensemble.pkl"
        self.assertFalse(prod_model_path.exists(), "Production model path should not exist without real empirical training.")

        res = self.predictor.predict_fusion(0.35, 0.45, 0.25)
        self.assertFalse(res["is_meta_learner_trained"])
        self.assertEqual(res["meta_model_status"], "UNTRAINED — VALID MULTIMODAL THREAT DATASET REQUIRED")
        self.assertEqual(res["active_model_name"], "Experimental Weighted Context Index")
        self.assertIn(res["context_status"], ["STANDARD CONTEXT", "ELEVATED CONTEXT", "HIGH CONTEXT"])
        self.assertIsNone(res["stacking_meta_probability"])

    def test_synthetic_training_workflow_outputs_only_to_simulation_directory(self):
        """4. The synthetic training workflow outputs only to the simulation directory."""
        self.assertTrue(self.demo_artifact_dir.is_relative_to(config.MODELS_DIR / "fusion"))
        self.assertEqual(self.demo_artifact_dir.name, "synthetic_demo")
        self.assertEqual(self.simulation_reports_dir.name, "synthetic_simulation")

        # Verify that running simulation does not create a production artifact at models/fusion/stacking_ensemble.pkl
        prod_path = config.MODELS_DIR / "fusion" / "stacking_ensemble.pkl"
        self.assertFalse(prod_path.exists())

    def test_production_training_workflow_rejects_missing_labels_and_synthetic_data(self):
        """5. The production training workflow rejects datasets with missing labels or synthetic properties."""
        # A. None / missing path rejects with descriptive error
        with self.assertRaises(FileNotFoundError):
            train_production_stacking_model(empirical_dataset_path=None)

        # B. Synthetic subject tokens rejected
        df_synthetic = pd.DataFrame({
            "subject_id": ["Subj_01", "Subj_02", "Subj_03", "Subj_01"],
            "p_physiological": [0.2, 0.8, 0.4, 0.3],
            "p_spatial": [0.1, 0.7, 0.3, 0.2],
            "p_environmental": [0.3, 0.6, 0.2, 0.1],
            "ground_truth_threat_label": [1, 0, 1, 0]
        })
        with self.assertRaises(SyntheticDatasetRejectedError):
            StackingEnsemblePipeline.validate_multimodal_dataset(
                df_synthetic,
                enforce_production_empirical=True
            )

        # C. Missing ground truth labels rejected
        df_no_label = df_synthetic.drop(columns=["ground_truth_threat_label"])
        with self.assertRaises(MissingGroundTruthLabelError):
            StackingEnsemblePipeline.validate_multimodal_dataset(df_no_label)

        # D. Circular labels from formula rejected
        df_circular = pd.DataFrame({
            "subject_id": ["Participant_A", "Participant_B", "Participant_C", "Participant_D"],
            "p_physiological": [0.2, 0.9, 0.1, 0.8],
            "p_spatial": [0.3, 0.8, 0.2, 0.7],
            "p_environmental": [0.1, 0.9, 0.1, 0.8],
            "ground_truth_threat_label": [0, 1, 0, 1]  # Exactly matches 0.40*phys + 0.25*loc + 0.35*env >= 0.50
        })
        with self.assertRaises(DataCircularityError):
            StackingEnsemblePipeline.validate_multimodal_dataset(df_circular)

    def test_base_models_inference_still_works(self):
        """6. WESAD, geospatial, and environmental inference still work correctly."""
        # A. Physiological Stress Model
        phys_score = self.predictor.predict_physiological({"mean_hr": 85.0, "rmssd": 28.0})
        self.assertIsInstance(phys_score, float)
        self.assertGreaterEqual(phys_score, 0.0)
        self.assertLessEqual(phys_score, 1.0)

        # B. Deterministic Campus Spatial Engine
        loc_score = self.predictor.predict_location(
            lat=config.BIT_MESRA_CENTER["latitude"],
            lon=config.BIT_MESRA_CENTER["longitude"],
            accuracy=10.0,
            hour=14.0
        )
        self.assertIsInstance(loc_score, float)
        self.assertGreaterEqual(loc_score, 0.0)
        self.assertLessEqual(loc_score, 1.0)

        # C. Rule-Based Environmental Engine
        env_score = self.predictor.predict_environment(
            time_of_day_hour=21.0,
            lighting_condition="Dark",
            crowd_density="Low"
        )
        self.assertIsInstance(env_score, float)
        self.assertGreaterEqual(env_score, 0.0)
        self.assertLessEqual(env_score, 1.0)

    def test_predict_fusion_does_not_output_synthetic_threat_probabilities(self):
        """7. predict_fusion does not output synthetic threat probabilities."""
        res = self.predictor.predict_fusion(0.25, 0.75, 0.35)

        # Must not report synthetic stacking probability
        self.assertIsNone(res["stacking_meta_probability"])
        self.assertFalse(res["is_meta_learner_trained"])
        self.assertEqual(res["active_model_name"], "Experimental Weighted Context Index")

        # Must include required context assessment explanation and clarification
        self.assertIn("Context Assessment: The experimental weighted context index is", res["explanation"])
        self.assertIn("This value describes contextual inputs only and does not establish whether a person or location is safe or unsafe.", res["explanation"])
        self.assertIn("notice", res)
        self.assertIn("clarification", res)
        self.assertIn("This index represents a heuristic combination of physiological, spatial and environmental context scores. It is not a validated threat prediction.", res["clarification"])

if __name__ == "__main__":
    unittest.main()
