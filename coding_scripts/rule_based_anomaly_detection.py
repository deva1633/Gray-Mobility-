import pandas as pd
import numpy as np

# -----------------------------
# LOAD FEATURE DATA
# -----------------------------
df = pd.read_csv("data/processed/features.csv")

# -----------------------------
# PARAMETERS (EXPLICIT & DEFENSIBLE)
# -----------------------------

# HR (early warning)
HR_SLOPE_THRESHOLD = 0.15        # bpm/sec
HR_PERSIST_SEC = 30              # seconds

# SpO2 (confirmation)
SPO2_DELTA_THRESHOLD = -3.0      # %
SPO2_PERSIST_SEC_1 = 30
SPO2_PERSIST_SEC_2 = 60

# BP (severity)
BP_SLOPE_THRESHOLD = -0.05       # mmHg/sec
BP_PERSIST_SEC = 60

# -----------------------------
# OUTPUT COLUMNS
# -----------------------------
df["hr_anomaly"] = 0
df["spo2_anomaly"] = 0
df["bp_anomaly"] = 0
df["anomaly_level"] = 0
df["reason"] = "normal"

# -----------------------------
# SLIDING WINDOW RULE ENGINE
# -----------------------------
for i in range(len(df)):

    # -------------------------
    # HR ANOMALY (TREND-BASED)
    # -------------------------
    hr_window = df.iloc[max(0, i - HR_PERSIST_SEC):i]

    if len(hr_window) >= HR_PERSIST_SEC:
        if (
            (hr_window["hr_slope_30s"] > HR_SLOPE_THRESHOLD).all()
            and df.loc[i, "high_motion_flag"] == 0
        ):
            df.loc[i, "hr_anomaly"] = 1

    # -------------------------
    # SpO2 ANOMALY (PERSISTENCE)
    # -------------------------
    if (
        (df.loc[i, "spo2_delta_from_baseline"] < SPO2_DELTA_THRESHOLD
         and df.loc[i, "spo2_seconds_below_94"] >= SPO2_PERSIST_SEC_1)
        or
        (df.loc[i, "spo2_seconds_below_94"] >= SPO2_PERSIST_SEC_2)
    ):
        df.loc[i, "spo2_anomaly"] = 1

    # -------------------------
    # BP ANOMALY (SEVERITY)
    # -------------------------
    bp_window = df.iloc[max(0, i - BP_PERSIST_SEC):i]

    if len(bp_window) >= BP_PERSIST_SEC:
        if (bp_window["sys_bp_slope_60s"] < BP_SLOPE_THRESHOLD).all():
            df.loc[i, "bp_anomaly"] = 1

    # -------------------------
    # FUSION LOGIC
    # -------------------------
    if df.loc[i, "hr_anomaly"] == 1:
        df.loc[i, "anomaly_level"] = 1
        df.loc[i, "reason"] = "HR rising trend"

    if df.loc[i, "hr_anomaly"] == 1 and df.loc[i, "spo2_anomaly"] == 1:
        df.loc[i, "anomaly_level"] = 2
        df.loc[i, "reason"] = "HR rise + sustained SpO2 drop"

    if (
        df.loc[i, "hr_anomaly"] == 1
        and df.loc[i, "spo2_anomaly"] == 1
        and df.loc[i, "bp_anomaly"] == 1
    ):
        df.loc[i, "anomaly_level"] = 3
        df.loc[i, "reason"] = "HR + SpO2 + BP deterioration"

# -----------------------------
# SAVE OUTPUT
# -----------------------------
df.to_csv("data/risk_scores/rule_based_anomaly_output.csv", index=False)

print("Rule-based anomaly detection complete.")
print(df["anomaly_level"].value_counts())
