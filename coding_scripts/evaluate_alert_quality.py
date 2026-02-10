import pandas as pd
import numpy as np

# =========================================================
# LOAD DATA
# =========================================================
df = pd.read_csv("data/risk_scores/final_decision.csv")

# =========================================================
# GROUND TRUTH (PHYSIOLOGY-BASED, REALISTIC)
# =========================================================
# Represents clinically meaningful deterioration
df["ground_truth"] = (
    (df["heart_rate_bpm"] > 100) |
    (df["spo2_percent"] < 94)
).astype(int)

# =========================================================
# ALERT POLICY (OPTIMIZED & BALANCED)
# =========================================================

# AMBER flag
df["amber_flag"] = (df["risk_level"] == "AMBER").astype(int)

# Sustained AMBER for 50 seconds (5 × 10s)
df["amber_sustained_50s"] = (
    df["amber_flag"]
    .rolling(window=5, min_periods=5)
    .mean()
    .fillna(0) == 1
).astype(int)

# --- Fast AMBER path (earlier detection, controlled) ---
amber_fast = (
    (df["risk_level"] == "AMBER") &
    (df["risk_score"] > 55)
)

# --- Slow AMBER path (sustained deterioration) ---
amber_slow = (
    (df["amber_sustained_50s"] == 1) &
    (df["risk_score"] > 45)
)

# --- RED alerts (always trigger) ---
red_alert = (df["risk_level"] == "RED")

# Final alert decision
df["predicted_alert"] = (
    red_alert |
    amber_fast |
    amber_slow
).astype(int)

# =========================================================
# CONFUSION MATRIX
# =========================================================
TP = ((df["predicted_alert"] == 1) & (df["ground_truth"] == 1)).sum()
FP = ((df["predicted_alert"] == 1) & (df["ground_truth"] == 0)).sum()
FN = ((df["predicted_alert"] == 0) & (df["ground_truth"] == 1)).sum()
TN = ((df["predicted_alert"] == 0) & (df["ground_truth"] == 0)).sum()

# =========================================================
# METRICS
# =========================================================
precision = TP / (TP + FP + 1e-6)
recall = TP / (TP + FN + 1e-6)
accuracy = (TP + TN) / (TP + TN + FP + FN + 1e-6)
false_alert_rate = FP / len(df)

# =========================================================
# ALERT LATENCY (CORRECT & MEANINGFUL)
# =========================================================
# Time from first ground-truth deterioration to first alert
gt_times = df.loc[df["ground_truth"] == 1, "time_sec"]
alert_times = df.loc[df["predicted_alert"] == 1, "time_sec"]

if not gt_times.empty and not alert_times.empty:
    alert_latency = alert_times.min() - gt_times.min()
else:
    alert_latency = np.nan

# =========================================================
# OUTPUT
# =========================================================
print("\n=== Alert Quality Metrics (FINAL OPTIMIZED) ===")
print(f"Precision        : {precision:.3f}")
print(f"Recall           : {recall:.3f}")
print(f"Accuracy         : {accuracy:.3f}")
print(f"False Alert Rate : {false_alert_rate:.4f}")
print(f"Alert Latency(s) : {alert_latency}")

print("\nAlert Rate (% of time):")
print(df['predicted_alert'].mean() * 100)
