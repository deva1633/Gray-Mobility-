import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os 
# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv("data/raw/synthetic_ambulance_vitals.csv")

time = df["time_sec"] / 60  # minutes

# Phase boundaries (minutes)
NORMAL_END = 10
DISTRESS_END = 20

# Keep a copy for comparison
raw = df.copy()

# -----------------------------
# PARAMETERS (EXPLICIT & DEFENSIBLE)
# -----------------------------
MOTION_THRESHOLD = 0.7
SPO2_DROP_THRESHOLD = 3          # %
HR_SPIKE_THRESHOLD = 15          # bpm
ROLLING_WINDOW = 5               # seconds

# -----------------------------
# HANDLE MISSING DATA
# -----------------------------
df["heart_rate_bpm"] = df["heart_rate_bpm"].interpolate(limit=5)
df["spo2_percent"] = df["spo2_percent"].interpolate(limit=5)
df["bp_systolic"] = df["bp_systolic"].interpolate(limit=5)
df["bp_diastolic"] = df["bp_diastolic"].interpolate(limit=5)

# -----------------------------
# ROLLING BASELINES
# -----------------------------
hr_baseline = df["heart_rate_bpm"].rolling(ROLLING_WINDOW, center=True).median()
spo2_baseline = df["spo2_percent"].rolling(ROLLING_WINDOW, center=True).median()

# -----------------------------
# HR SPIKE SUPPRESSION (motion-related)
# -----------------------------
hr_spike_mask = (
    (df["motion"] > MOTION_THRESHOLD) &
    ((df["heart_rate_bpm"] - hr_baseline).abs() > HR_SPIKE_THRESHOLD)
)

df.loc[hr_spike_mask, "heart_rate_bpm"] = hr_baseline[hr_spike_mask]

# -----------------------------
# SpO2 FALSE DROP SUPPRESSION
# -----------------------------
spo2_artifact_mask = (
    (df["motion"] > MOTION_THRESHOLD) &
    ((spo2_baseline - df["spo2_percent"]) > SPO2_DROP_THRESHOLD)
)

df.loc[spo2_artifact_mask, "spo2_percent"] = spo2_baseline[spo2_artifact_mask]

# -----------------------------
# BEFORE vs AFTER PLOTS
# -----------------------------

# HEART RATE
plt.figure()
plt.plot(time, raw["heart_rate_bpm"], label="Raw", alpha=0.6)
plt.plot(time, df["heart_rate_bpm"], label="After Artifact Handling")
plt.axvline(NORMAL_END, linestyle="--")
plt.axvline(DISTRESS_END, linestyle="--")
plt.title("Heart Rate — Before vs After Artifact Handling")
plt.xlabel("Time (minutes)")
plt.ylabel("HR (bpm)")
plt.legend()
plt.show()

# SpO2
plt.figure()
plt.plot(time, raw["spo2_percent"], label="Raw", alpha=0.6)
plt.plot(time, df["spo2_percent"], label="After Artifact Handling")
plt.axvline(NORMAL_END, linestyle="--")
plt.axvline(DISTRESS_END, linestyle="--")
plt.title("SpO₂ — Before vs After Artifact Handling")
plt.xlabel("Time (minutes)")
plt.ylabel("SpO₂ (%)")
plt.legend()
plt.show()

# -----------------------------
# SAVE CLEANED DATA
# -----------------------------
os.makedirs("data/processed", exist_ok=True)
df.to_csv("data/processed/synthetic_ambulance_vitals_cleaned.csv", index=False)
print("Artifact handling complete. Cleaned dataset saved.")