import json
import pandas as pd
from pathlib import Path

from pathlib import Path

# -------------------------
# PATH SETUP (CORRECT)
# -------------------------
BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data" / "features.csv"
STATS_PATH = BASE_DIR / "models" / "training_stats.json"
DRIFT_TEST_PATH = BASE_DIR / "data" / "drift_test.csv"


# -------------------------
# DRIFT DETECTION FUNCTION
# -------------------------
def detect_drift(data_path, threshold=2.5):
    df = pd.read_csv(data_path)

    with open(STATS_PATH, "r") as f:
        stats = json.load(f)

    drift_features = []

    for col in df.columns:
        if col in stats["mean"]:
            z_score = abs(
                (df[col].mean() - stats["mean"][col]) /
                (stats["std"][col] + 1e-8)
            )
            if z_score > threshold:
                drift_features.append(col)

    return drift_features


# -------------------------
# MAIN EXECUTION
# -------------------------
if __name__ == "__main__":

    # 1️⃣ Normal drift check (no artificial drift)
    print("🔍 Checking drift on original data...")
    drifted = detect_drift(DATA_PATH)

    if drifted:
        print("⚠️ Drift detected in:", drifted)
    else:
        print("✅ No significant drift detected")

    # 2️⃣ Inject artificial drift for validation
    print("\n🧪 Injecting artificial drift for validation...")

    df = pd.read_csv(DATA_PATH)
    df_drifted = df.copy()

    # amplify first feature to simulate drift
    df_drifted.iloc[:, 0] = df_drifted.iloc[:, 0] * 3

    df_drifted.to_csv(DRIFT_TEST_PATH, index=False)

    drifted_test = detect_drift(DRIFT_TEST_PATH)

    if drifted_test:
        print("⚠️ Drift detected after injection in:", drifted_test)
    else:
        print("❌ Drift NOT detected (check threshold)")
