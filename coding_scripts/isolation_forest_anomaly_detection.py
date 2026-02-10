import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# -----------------------------
# LOAD FEATURE DATA
# -----------------------------
df = pd.read_csv("data/processed/features.csv")

# -----------------------------
# SELECT FEATURES FOR IF
# (engineered, interpretable features only)
# -----------------------------
feature_cols = [
    "hr_mean_30s",
    "hr_slope_30s",
    "hr_std_30s",
    "spo2_mean_30s",
    "spo2_delta_from_baseline",
    "spo2_seconds_below_94",
    "sys_bp_mean_60s",
    "sys_bp_slope_60s",
    "motion_mean_10s"
]

X = df[feature_cols].copy()

# -----------------------------
# SCALE FEATURES
# -----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------
# TRAIN ISOLATION FOREST
# -----------------------------
iso_forest = IsolationForest(
    n_estimators=200,
    contamination=0.08,   # expected anomaly proportion
    random_state=42
)

iso_forest.fit(X_scaled)

# -----------------------------
# ANOMALY SCORES & FLAGS
# -----------------------------
# Higher score = more anomalous (invert sklearn convention)
df["if_score"] = -iso_forest.score_samples(X_scaled)

# Binary flag (1 = anomaly)
df["if_anomaly"] = (iso_forest.predict(X_scaled) == -1).astype(int)

# -----------------------------
# NORMALIZE SCORE (0–1)
# -----------------------------
df["if_score_norm"] = (
    (df["if_score"] - df["if_score"].min())
    / (df["if_score"].max() - df["if_score"].min())
)

# -----------------------------
# SAVE OUTPUT
# -----------------------------
df.to_csv("data/risk_scores/isolation_forest_output.csv", index=False)

print("Isolation Forest anomaly detection complete.")
print(df["if_anomaly"].value_counts())
