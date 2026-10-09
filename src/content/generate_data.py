import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Ensure project root is in sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
import config

def generate_content_dataset(
    num_samples: int = 1800,
    output_csv: Path = config.DATA_DIR / "content" / "content_data.csv",
    seed: int = 42
) -> pd.DataFrame:
    """
    Generates synthetic distress/threat communication text samples for Model 4.
    """
    np.random.seed(seed)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    
    benign_phrases = [
        "Hey, running a bit late, see you soon!",
        "Just finished work, heading to the subway.",
        "Stuck in traffic on Main Street, will be 10 mins.",
        "Can you send me the address for dinner?",
        "Great movie tonight! Thanks for organizing.",
        "On my way back home, everything looks good.",
        "Checking in, all safe here.",
        "Walking to the store to get groceries.",
        "Battery is at 50%, call you when I reach.",
        "Having coffee with friends near downtown.",
        "Met up with the group, talk later!",
        "Train is running on schedule today."
    ]
    
    threat_phrases = [
        "HELP ME! Someone is following me down this dark road!",
        "Emergency! Suspicious person behind me near 4th street, I am scared.",
        "SOS! Call police immediately, I feel trapped!",
        "Someone is aggressive and following my steps, please hurry!",
        "I am being chased, need emergency help NOW!",
        "Unsafe situation in park, strange man approaching me fast.",
        "Distress alert! Car following me for 3 blocks.",
        "Help! I hear footsteps right behind me in the alley.",
        "Emergency broadcast: trapped in parking structure, need security!",
        "Someone is trying to grab my bag, call 911!",
        "URGENT: unsafe area, someone stalking me, send location to police!"
    ]
    
    rows = []
    half = num_samples // 2
    
    for _ in range(half):
        phrase = np.random.choice(benign_phrases)
        rows.append({"text_content": phrase, "content_risk_label": 0})
        
    for _ in range(half):
        phrase = np.random.choice(threat_phrases)
        rows.append({"text_content": phrase, "content_risk_label": 1})
        
    df = pd.DataFrame(rows).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    df.to_csv(output_csv, index=False)
    
    print(f"Content dataset generated with shape {df.shape} -> Saved to: {output_csv}", flush=True)
    print(f"Target Distribution: Class 0 (Benign): {np.sum(df['content_risk_label']==0)}, Class 1 (Threat): {np.sum(df['content_risk_label']==1)}", flush=True)
    return df

if __name__ == "__main__":
    generate_content_dataset()
