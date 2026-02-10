import pandas as pd
import numpy as np
import os

# =========================================================
# CONFIG
# =========================================================
MAX_ACCEPTABLE_DELAY = 180  # seconds (clinical response window)

# =========================================================
# LOAD DATA
# =========================================================
df = pd.read_csv("data/risk_scores/final_decision.csv")
os.makedirs("analysis/failure_cases", exist_ok=True)

# =========================================================
# GROUND TRUTH
# =========================================================
df["ground_truth"] = (
    (df["heart_rate_bpm"] > 100) |
    (df["spo2_percent"] < 94)
).astype(int)

# =========================================================
# RECOMPUTE ALERT LOGIC (FINAL POLICY)
# =========================================================
df["amber_flag"] = (df["risk_level"] == "AMBER").astype(int)

df["amber_sustained_50s"] = (
    df["amber_flag"]
    .rolling(window=5, min_periods=5)
    .mean()
    .fillna(0) == 1
).astype(int)

amber_fast = (
    (df["risk_level"] == "AMBER") &
    (df["risk_score"] > 55)
)

amber_slow = (
    (df["amber_sustained_50s"] == 1) &
    (df["risk_score"] > 45)
)

red_alert = (df["risk_level"] == "RED")

df["predicted_alert"] = (
    red_alert | amber_fast | amber_slow
).astype(int)

# =========================================================
# EVENT STARTS
# =========================================================
df["gt_start"] = (df["ground_truth"] == 1) & (df["ground_truth"].shift(1) == 0)
df["alert_start"] = (df["predicted_alert"] == 1) & (df["predicted_alert"].shift(1) == 0)

# =========================================================
# FAILURE CASE 1: TRANSIENT FALSE POSITIVES
# =========================================================
fp_transient = df[
    (df["predicted_alert"] == 1) &
    (df["ground_truth"] == 0)
].head(30)

fp_transient.to_csv(
    "analysis/failure_cases/failure_case_1_transient_false_positive.csv",
    index=False
)

# =========================================================
# FAILURE CASE 2: LATE ALERTS
# =========================================================
late_alert_events = []

for gt_idx in df.index[df["gt_start"]]:
    gt_time = df.loc[gt_idx, "time_sec"]

    future_alerts = df[
        (df["time_sec"] > gt_time) &
        (df["predicted_alert"] == 1)
    ]

    if not future_alerts.empty:
        first_alert_time = future_alerts.iloc[0]["time_sec"]
        latency = first_alert_time - gt_time

        if latency > MAX_ACCEPTABLE_DELAY:
            event = df.loc[gt_idx].copy()
            event["alert_latency_sec"] = latency
            late_alert_events.append(event)

late_alert_df = pd.DataFrame(late_alert_events)

late_alert_df.to_csv(
    "analysis/failure_cases/failure_case_2_late_alert.csv",
    index=False
)

# =========================================================
# FAILURE CASE 3: FALSE NEGATIVES (TIME-BOUNDED)
# =========================================================
false_negative_events = []

for gt_idx in df.index[df["gt_start"]]:
    gt_time = df.loc[gt_idx, "time_sec"]

    alerts_in_window = df[
        (df["time_sec"] > gt_time) &
        (df["time_sec"] <= gt_time + MAX_ACCEPTABLE_DELAY) &
        (df["predicted_alert"] == 1)
    ]

    if alerts_in_window.empty:
        false_negative_events.append(df.loc[gt_idx])

false_negative_df = pd.DataFrame(false_negative_events)

false_negative_df.to_csv(
    "analysis/failure_cases/failure_case_3_false_negative_time_bounded.csv",
    index=False
)

# =========================================================
# SUMMARY
# =========================================================
print("\n=== FAILURE ANALYSIS SUMMARY (TIME-BOUNDED) ===")
print(f"Transient False Positives : {len(fp_transient)}")
print(f"Late Alert Events         : {len(late_alert_df)}")
print(f"False Negative Events     : {len(false_negative_df)}")
print(f"Time Window (seconds)     : {MAX_ACCEPTABLE_DELAY}")
