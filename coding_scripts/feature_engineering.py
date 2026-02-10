import pandas as pd
import numpy as np

# -----------------------------
# LOAD CLEANED DATA
# -----------------------------
df = pd.read_csv("data/processed/synthetic_ambulance_vitals_cleaned.csv")

# -----------------------------
# WINDOW SIZES (seconds)
# -----------------------------
WIN_10 = 10
WIN_30 = 30
WIN_60 = 60

# -----------------------------
# HELPER: SLOPE CALCULATION
# -----------------------------
def rolling_slope(series, window):
    return series.rolling(window).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == window else np.nan,
        raw=False
    )

# -----------------------------
# HEART RATE FEATURES
# -----------------------------
df["hr_mean_30s"] = df["heart_rate_bpm"].rolling(WIN_30).mean()
df["hr_std_30s"] = df["heart_rate_bpm"].rolling(WIN_30).std()
df["hr_slope_30s"] = rolling_slope(df["heart_rate_bpm"], WIN_30)
# -----------------------------
# SpO2 FEATURES (REDESIGNED)
# -----------------------------

# Rolling mean (primary SpO2 signal)
df["spo2_mean_30s"] = df["spo2_percent"].rolling(WIN_30).mean()

# Baseline SpO2 (assume first 5 minutes are stable)
BASELINE_WINDOW = 5 * 60  # seconds
spo2_baseline = df["spo2_percent"].iloc[:BASELINE_WINDOW].mean()

# Deviation from baseline (captures sustained hypoxia)
df["spo2_delta_from_baseline"] = df["spo2_mean_30s"] - spo2_baseline

# Persistence below clinical threshold (robust to noise)
SPO2_THRESHOLD = 94
PERSIST_WINDOW = 60  # seconds

df["spo2_seconds_below_94"] = (
    (df["spo2_percent"] < SPO2_THRESHOLD)
    .rolling(PERSIST_WINDOW)
    .sum()
)

# OPTIONAL: keep slope as a weak / auxiliary signal only
df["spo2_slope_60s"] = rolling_slope(df["spo2_percent"], 60)




# -----------------------------
# BLOOD PRESSURE FEATURES
# -----------------------------
df["sys_bp_mean_60s"] = df["bp_systolic"].rolling(WIN_60).mean()
df["sys_bp_slope_60s"] = rolling_slope(df["bp_systolic"], WIN_60)

# -----------------------------
# MOTION CONTEXT FEATURES
# -----------------------------
df["motion_mean_10s"] = df["motion"].rolling(WIN_10).mean()
df["high_motion_flag"] = (df["motion_mean_10s"] > 0.7).astype(int)

# -----------------------------
# DROP INITIAL NaNs
# -----------------------------
feature_df = df.dropna().reset_index(drop=True)

# -----------------------------
# SAVE FEATURES
# -----------------------------
feature_df.to_csv("data/processed/features.csv", index=False)

print("Feature engineering complete. Features saved to data/processed/features.csv")
print(feature_df.head())
