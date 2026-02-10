import pandas as pd
import matplotlib.pyplot as plt
import os

# -------------------------
# CONFIG
# -------------------------
INPUT_PATH = "data/ambulance_vitals_dataset.csv"
OUTPUT_PATH = "data/cleaned_vitals.csv"

os.makedirs("data", exist_ok=True)

# -------------------------
# LOAD DATA
# -------------------------
df = pd.read_csv(INPUT_PATH)

# -------------------------
# ARTIFACT HANDLING
# -------------------------

# 1. Interpolate missing HR and SpO2 per patient
df["HR"] = df.groupby("patient_id")["HR"].transform(lambda x: x.interpolate())
df["SpO2"] = df.groupby("patient_id")["SpO2"].transform(lambda x: x.interpolate())

# 2. Rolling median smoothing
df["HR_clean"] = df.groupby("patient_id")["HR"].transform(
    lambda x: x.rolling(window=5, center=True).median()
)

df["SpO2_clean"] = df.groupby("patient_id")["SpO2"].transform(
    lambda x: x.rolling(window=5, center=True).median()
)

# Fill edges
df["HR_clean"] = df["HR_clean"].fillna(df["HR"])
df["SpO2_clean"] = df["SpO2_clean"].fillna(df["SpO2"])


# 3. Motion-aware artifact suppression
high_motion = df["motion"] > 0.9   # 0.6 

df.loc[high_motion, "SpO2_clean"] = (
    df.groupby("patient_id")["SpO2_clean"]
    .transform(lambda x: x.rolling(10, min_periods=1).median())
)

# -------------------------
# SAVE CLEANED DATA
# -------------------------
df.to_csv(OUTPUT_PATH, index=False)
print("Cleaned dataset saved to:", OUTPUT_PATH)

# -------------------------
# BEFORE vs AFTER PLOTS
# -------------------------
patient = df[df["patient_id"] == 1]

plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(patient["timestamp"], patient["HR"], label="Raw HR", alpha=0.5)
plt.plot(patient["timestamp"], patient["HR_clean"], label="Clean HR")
plt.legend()
plt.title("HR Artifact Cleaning")

plt.subplot(2, 1, 2)
plt.plot(patient["timestamp"], patient["SpO2"], label="Raw SpO2", alpha=0.5)
plt.plot(patient["timestamp"], patient["SpO2_clean"], label="Clean SpO2")
plt.legend()
plt.title("SpO2 Artifact Cleaning")

plt.tight_layout()
plt.show()
