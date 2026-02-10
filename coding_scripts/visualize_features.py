import os
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# SETUP
# -----------------------------
os.makedirs("plots/features", exist_ok=True)

df = pd.read_csv("data/features.csv")

time = df["time_sec"] / 60  # minutes

NORMAL_END = 10
DISTRESS_END = 20

# =========================================================
# HEART RATE FEATURES
# =========================================================
fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 6))

axes[0].plot(time, df["hr_mean_30s"])
axes[0].axvline(NORMAL_END, linestyle="--")
axes[0].axvline(DISTRESS_END, linestyle="--")
axes[0].set_title("Heart Rate — Rolling Mean (30s)")
axes[0].set_ylabel("HR (bpm)")

axes[1].plot(time, df["hr_slope_30s"])
axes[1].axvline(NORMAL_END, linestyle="--")
axes[1].axvline(DISTRESS_END, linestyle="--")
axes[1].set_title("Heart Rate — Rolling Slope (30s)")
axes[1].set_xlabel("Time (minutes)")
axes[1].set_ylabel("HR slope (bpm/sec)")

plt.tight_layout()
plt.savefig("plots/features/hr_features.png", dpi=150)
plt.close()

# =========================================================
# SpO₂ FEATURES (REDESIGNED)
# =========================================================
fig, axes = plt.subplots(3, 1, sharex=True, figsize=(10, 8))

axes[0].plot(time, df["spo2_mean_30s"])
axes[0].axvline(NORMAL_END, linestyle="--")
axes[0].axvline(DISTRESS_END, linestyle="--")
axes[0].set_title("SpO₂ — Rolling Mean (30s)")
axes[0].set_ylabel("SpO₂ (%)")

axes[1].plot(time, df["spo2_delta_from_baseline"])
axes[1].axhline(0, linestyle="--")
axes[1].axvline(NORMAL_END, linestyle="--")
axes[1].axvline(DISTRESS_END, linestyle="--")
axes[1].set_title("SpO₂ — Deviation from Baseline")
axes[1].set_ylabel("Δ SpO₂ (%)")

axes[2].plot(time, df["spo2_seconds_below_94"])
axes[2].axhline(30, linestyle="--", label="30s")
axes[2].axhline(60, linestyle="--", label="60s")
axes[2].axvline(NORMAL_END, linestyle="--")
axes[2].axvline(DISTRESS_END, linestyle="--")
axes[2].set_title("SpO₂ — Persistence Below 94%")
axes[2].set_xlabel("Time (minutes)")
axes[2].set_ylabel("Seconds")
axes[2].legend()

plt.tight_layout()
plt.savefig("plots/features/spo2_features.png", dpi=150)
plt.close()

# =========================================================
# BLOOD PRESSURE FEATURES
# =========================================================
fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 6))

axes[0].plot(time, df["sys_bp_mean_60s"])
axes[0].axvline(NORMAL_END, linestyle="--")
axes[0].axvline(DISTRESS_END, linestyle="--")
axes[0].set_title("Systolic BP — Rolling Mean (60s)")
axes[0].set_ylabel("BP (mmHg)")

axes[1].plot(time, df["sys_bp_slope_60s"])
axes[1].axvline(NORMAL_END, linestyle="--")
axes[1].axvline(DISTRESS_END, linestyle="--")
axes[1].set_title("Systolic BP — Rolling Slope (60s)")
axes[1].set_xlabel("Time (minutes)")
axes[1].set_ylabel("BP slope (mmHg/sec)")

plt.tight_layout()
plt.savefig("plots/features/bp_features.png", dpi=150)
plt.close()

# =========================================================
# MOTION CONTEXT FEATURES
# =========================================================
fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 6))

axes[0].plot(time, df["motion_mean_10s"])
axes[0].axvline(NORMAL_END, linestyle="--")
axes[0].axvline(DISTRESS_END, linestyle="--")
axes[0].set_title("Motion — Rolling Mean (10s)")
axes[0].set_ylabel("Motion")

axes[1].step(time, df["high_motion_flag"])
axes[1].axvline(NORMAL_END, linestyle="--")
axes[1].axvline(DISTRESS_END, linestyle="--")
axes[1].set_title("High Motion Flag")
axes[1].set_xlabel("Time (minutes)")
axes[1].set_ylabel("Flag (0/1)")

plt.tight_layout()
plt.savefig("plots/features/motion_features.png", dpi=150)
plt.close()

print("Feature visualizations saved successfully to plots/features/")
