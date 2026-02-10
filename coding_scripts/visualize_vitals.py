import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv("data/synthetic_ambulance_vitals.csv")

time = df["time_sec"] / 60  # convert to minutes for readability

# Phase boundaries (minutes)
NORMAL_END = 10
DISTRESS_END = 20

# -----------------------------
# HEART RATE
# -----------------------------
plt.figure()
plt.plot(time, df["heart_rate_bpm"])
plt.axvline(NORMAL_END, linestyle="--")
plt.axvline(DISTRESS_END, linestyle="--")
plt.title("Heart Rate Over Time")
plt.xlabel("Time (minutes)")
plt.ylabel("HR (bpm)")
plt.show()

# -----------------------------
# SpO2
# -----------------------------
plt.figure()
plt.plot(time, df["spo2_percent"])
plt.axvline(NORMAL_END, linestyle="--")
plt.axvline(DISTRESS_END, linestyle="--")
plt.title("SpO₂ Over Time")
plt.xlabel("Time (minutes)")
plt.ylabel("SpO₂ (%)")
plt.show()

# -----------------------------
# BLOOD PRESSURE
# -----------------------------
plt.figure()
plt.plot(time, df["bp_systolic"], label="Systolic")
plt.plot(time, df["bp_diastolic"], label="Diastolic")
plt.axvline(NORMAL_END, linestyle="--")
plt.axvline(DISTRESS_END, linestyle="--")
plt.title("Blood Pressure Over Time")
plt.xlabel("Time (minutes)")
plt.ylabel("BP (mmHg)")
plt.legend()
plt.show()

# -----------------------------
# MOTION SIGNAL
# -----------------------------
plt.figure()
plt.plot(time, df["motion"])
plt.axvline(NORMAL_END, linestyle="--")
plt.axvline(DISTRESS_END, linestyle="--")
plt.title("Motion / Vibration Signal")
plt.xlabel("Time (minutes)")
plt.ylabel("Motion (unitless)")
plt.show()
