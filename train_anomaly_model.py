import pandas as pd
import json
from pathlib import Path

# Get project root (one level above coding_scripts)
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "features.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA_PATH)

stats = {
    "mean": df.mean().to_dict(),
    "std": df.std().to_dict()
}

with open(MODEL_DIR / "training_stats.json", "w") as f:
    json.dump(stats, f)

print("✅ Training statistics saved")


