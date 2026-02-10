import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# SETUP
# -----------------------------
os.makedirs("plots/validation/artifacts", exist_ok=True)

raw = pd.read_csv("data/raw/synthetic_ambulance_vitals.csv")
clean = pd.read_csv("data/processed/synthetic_ambulance_vitals_cleaned.csv")

time = raw["time_sec"] / 60  # minutes

NORMAL_END = 10
DISTRESS_END = 20

# -----------------------------
# PARAMETERS (same as detection)
# -----------------------------
MOTION_THRESHOLD = 0.7
HR_SPIKE_THRESHOLD = 15
SPO2_DROP_THRESHOLD = 3
ROLLING_WINDOW = 5

# -----------------------------
# BASELINES (for artifact marking)
# -----------------------------
hr_baseline = clean["heart_rate_bpm"].rolling(ROLLING_WINDOW, center=True).median()
spo2_baseline = clean["spo2_percent"].rolling(ROLLING_WINDOW, center=True).median()

# -----------------------------
# ARTIFACT MASKS
# -----------------------------
hr_spike_mask = (
    (raw["motion"] > MOTION_THRESHOLD) &
    ((raw["heart_rate_bpm"] - hr_baseline).abs() > HR_SPIKE_THRESHOLD)
)

spo2_artifact_mask = (
    (raw["motion"] > MOTION_THRESHOLD) &
    ((spo2_baseline - raw["spo2_percent"]) > SPO2_DROP_THRESHOLD)
)

# =========================================================
# HEART RATE — BEFORE vs AFTER
# =========================================================
fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 6))

axes[0].plot(time, raw["heart_rate_bpm"])
axes[0].scatter(time[hr_spike_mask], raw["heart_rate_bpm"][hr_spike_mask], marker="x")
axes[0].axvline(NORMAL_END, linestyle="--")
axes[0].axvline(DISTRESS_END, linestyle="--")
axes[0].set_title("Heart Rate — Raw")
axes[0].set_ylabel("HR (bpm)")

axes[1].plot(time, clean["heart_rate_bpm"])
axes[1].axvline(NORMAL_END, linestyle="--")
axes[1].axvline(DISTRESS_END, linestyle="--")
axes[1].set_title("Heart Rate — After Artifact Handling")
axes[1].set_xlabel("Time (minutes)")
axes[1].set_ylabel("HR (bpm)")

plt.tight_layout()
plt.savefig("plots/validation/artifacts/hr_before_after.png", dpi=150)
plt.close()

# =========================================================
# SpO₂ — BEFORE vs AFTER
# =========================================================
fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 6))

axes[0].plot(time, raw["spo2_percent"])
axes[0].scatter(time[spo2_artifact_mask], raw["spo2_percent"][spo2_artifact_mask], marker="x")
axes[0].axvline(NORMAL_END, linestyle="--")
axes[0].axvline(DISTRESS_END, linestyle="--")
axes[0].set_title("SpO₂ — Raw")
axes[0].set_ylabel("SpO₂ (%)")

axes[1].plot(time, clean["spo2_percent"])
axes[1].axvline(NORMAL_END, linestyle="--")
axes[1].axvline(DISTRESS_END, linestyle="--")
axes[1].set_title("SpO₂ — After Artifact Handling")
axes[1].set_xlabel("Time (minutes)")
axes[1].set_ylabel("SpO₂ (%)")

plt.tight_layout()
plt.savefig("plots/validation/artifacts/spo2_before_after.png", dpi=150)
plt.close()

# =========================================================
# BLOOD PRESSURE — BEFORE vs AFTER
# =========================================================
fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 6))

# RAW BP
axes[0].plot(time, raw["bp_systolic"], label="Systolic")
axes[0].plot(time, raw["bp_diastolic"], label="Diastolic")
axes[0].axvline(NORMAL_END, linestyle="--")
axes[0].axvline(DISTRESS_END, linestyle="--")
axes[0].set_title("Blood Pressure — Raw")
axes[0].set_ylabel("BP (mmHg)")
axes[0].legend()

# CLEANED BP
axes[1].plot(time, clean["bp_systolic"], label="Systolic")
axes[1].plot(time, clean["bp_diastolic"], label="Diastolic")
axes[1].axvline(NORMAL_END, linestyle="--")
axes[1].axvline(DISTRESS_END, linestyle="--")
axes[1].set_title("Blood Pressure — After Artifact Handling")
axes[1].set_xlabel("Time (minutes)")
axes[1].set_ylabel("BP (mmHg)")
axes[1].legend()

# =========================================================
# MOTION / VIBRATION — BEFORE vs AFTER
# (Intentionally unchanged)
# =========================================================
fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 6))

# RAW motion
axes[0].plot(time, raw["motion"])
axes[0].axvline(NORMAL_END, linestyle="--")
axes[0].axvline(DISTRESS_END, linestyle="--")
axes[0].set_title("Motion / Vibration — Raw")
axes[0].set_ylabel("Motion (unitless)")

# AFTER (same as raw, intentionally)
axes[1].plot(time, clean["motion"])
axes[1].axvline(NORMAL_END, linestyle="--")
axes[1].axvline(DISTRESS_END, linestyle="--")
axes[1].set_title("Motion / Vibration — After Artifact Handling (Unchanged)")
axes[1].set_xlabel("Time (minutes)")
axes[1].set_ylabel("Motion (unitless)")

plt.tight_layout()
plt.savefig("plots/validation/artifacts/motion_before_after.png", dpi=150)
plt.close()


plt.tight_layout()
plt.savefig("plots/validation/artifacts/bp_before_after.png", dpi=150)
plt.close()

print("HR, SpO₂, and Blood Pressure before/after plots generated successfully.")
