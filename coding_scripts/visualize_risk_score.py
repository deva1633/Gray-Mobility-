import pandas as pd
import matplotlib.pyplot as plt
import os

# -------------------------
# PATHS
# -------------------------
INPUT_FILE = "data/risk_scores/final_decision.csv"
OUTPUT_DIR = "plots/risk_trends"
OUTPUT_IMAGE = os.path.join(OUTPUT_DIR, "risk_score_over_time.png")

# -------------------------
# CREATE OUTPUT FOLDER
# -------------------------
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------
# LOAD DATA
# -------------------------
df = pd.read_csv(INPUT_FILE)

# Sort by time for clean plot
df = df.sort_values("time_sec")

# -------------------------
# PLOT: RISK SCORE VS TIME
# -------------------------
plt.figure()
plt.plot(df["time_sec"], df["risk_score"])

# Threshold lines
plt.axhline(40, linestyle="--")
plt.axhline(70, linestyle="--")

plt.xlabel("Time (seconds)")
plt.ylabel("Risk Score")
plt.title("Risk Score Evolution Over Time")

# -------------------------
# SAVE FIGURE
# -------------------------
plt.savefig(OUTPUT_IMAGE, bbox_inches="tight")
plt.close()

print("✅ Risk score visualization saved")
print(f"📁 File location: {OUTPUT_IMAGE}")
