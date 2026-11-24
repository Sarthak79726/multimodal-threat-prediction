"""
End-to-end verification script testing complete application flow:
1. Streamlit local startup and interaction via AppTest.
2. Evaluate button presence, label, click execution, and results display.
3. 5 campus locations across various diurnal hours and weekday/weekend schedules.
4. Handling of unknown locations and missing inputs without fabricated data.
"""

import sys
from pathlib import Path
from streamlit.testing.v1 import AppTest

# Ensure project root is in sys.path
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import config
from src.fusion.predict import MultimodalThreatPredictor

def run_e2e_verification():
    print("=== 1. AppTest Headless Streamlit Verification ===")
    app_path = str(Path(_project_root) / "app" / "main.py")
    at = AppTest.from_file(app_path)
    at.run(timeout=20)
    assert len(at.exception) == 0, f"Startup failed with exceptions: {at.exception}"
    print("✓ Local Streamlit startup succeeded without errors.")

    # Verify Evaluate button presence and interaction
    btn = at.button(key="btn_main_threat")
    assert btn is not None
    print(f"✓ Evaluate button found with label: \"{btn.label}\"")

    btn.click().run(timeout=20)
    assert len(at.exception) == 0, f"Button click failed with exceptions: {at.exception}"
    threat = at.session_state["analyzed_threat"]
    assert threat is not None
    print("✓ Button click executed successfully.")
    print(f"  - Experimental Weighted Context Index: {threat['experimental_context_index_pct']:.2f}%")
    print(f"  - Context Classification: {threat['context_status']}")
    print(f"  - Meta-Learner Status: {threat['meta_model_status']}")

    print("\n=== 2. Multi-Location and Multi-Time Scenario Verification ===")
    p = MultimodalThreatPredictor()
    test_scenarios = [
        ("Main Building", 10.5, "Monday", 0),
        ("Girls Hostel Area / H8-H9", 21.0, "Wednesday", 0),
        ("PMC", 14.0, "Friday", 0),
        ("Sports Complex", 17.5, "Saturday", 1),
        ("Main Gate", 23.5, "Sunday", 1),
    ]

    for loc, hour, dow, is_wknd in test_scenarios:
        env_res = p.predict_environment_context(loc, hour, dow, is_wknd)
        loc_score = p.predict_location(hour=hour, loc_type="academic_zone" if "Building" in loc else "hostel_zone")
        phys_score = 0.22
        fusion = p.predict_fusion(phys_score, loc_score, env_res["calibrated_environmental_score"])
        
        print(f"Scenario: {loc} at {hour:.1f}h ({dow})")
        crowd_cat = env_res["crowd_prediction"]["category"]
        crowd_conf = env_res["crowd_prediction"]["confidence_pct"]
        light_cat = env_res["lighting_prediction"]["category"]
        light_conf = env_res["lighting_prediction"]["confidence_pct"]
        lux = env_res["lighting_prediction"].get("lux_estimate", 0)
        env_sc = env_res["calibrated_environmental_score"]
        final_pct = fusion["experimental_context_index_pct"]
        status = fusion["context_status"]

        print(f"  Crowd: {crowd_cat} ({crowd_conf}%) | Lighting: {light_cat} ({light_conf}%, {lux} lx)")
        print(f"  Scores: Phys={phys_score:.2f}, Loc={loc_score:.2f}, Env={env_sc:.2f}")
        print(f"  -> Final Index: {final_pct:.2f}% [{status}]")
        assert 0.0 <= fusion["experimental_context_index"] <= 1.0

    print("\n=== 3. Missing Inputs Policy Verification ===")
    unknown_env = p.predict_environment_context("Unknown Planet", 12.0)
    print("Unknown location status:", unknown_env["status"])
    print("Unknown crowd category:", unknown_env["crowd_prediction"]["category"])
    print("Unknown lighting category:", unknown_env["lighting_prediction"]["category"])
    fusion_unknown = p.predict_fusion(0.20, 0.15, unknown_env["calibrated_environmental_score"])
    print("Fusion with unknown env:", fusion_unknown["experimental_context_index_pct"], fusion_unknown["context_status"])
    assert unknown_env["status"] == "UNKNOWN"
    assert unknown_env["crowd_prediction"]["category"] == "UNKNOWN"
    assert fusion_unknown["experimental_context_index_pct"] == 18.75

    print("\n✓ ALL 12 END-TO-END VERIFICATION CRITERIA PASSED!")

if __name__ == "__main__":
    run_e2e_verification()
