import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# -----------------------------
# LOAD FEATURES
# -----------------------------
df = pd.read_csv("data/processed/features.csv")

# -----------------------------
# DEFINE NORMAL DATA (0–10 min, low motion)
# -----------------------------
time_min = df["time_sec"] / 60
normal_mask = (time_min < 10) & (df["high_motion_flag"] == 0)

# -----------------------------
# FEATURE SET
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

X = df[feature_cols]
X_normal = X[normal_mask]

# -----------------------------
# SCALE FEATURES
# -----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_normal_scaled = scaler.transform(X_normal)

# -----------------------------
# ISOLATION FOREST (NORMAL-ONLY)
# -----------------------------
iso_forest = IsolationForest(
    n_estimators=300,
    contamination=0.05,
    random_state=42
)
iso_forest.fit(X_normal_scaled)

df["if_score"] = -iso_forest.score_samples(X_scaled)

# Normalize IF score
df["if_score_norm"] = (
    (df["if_score"] - df["if_score"].min())
    / (df["if_score"].max() - df["if_score"].min())
)

# -----------------------------
# PCA (NORMAL-ONLY)
# -----------------------------
pca = PCA(n_components=0.95, random_state=42)
X_pca_normal = pca.fit_transform(X_normal_scaled)
X_pca_all = pca.transform(X_scaled)

X_recon = pca.inverse_transform(X_pca_all)
recon_error = np.mean((X_scaled - X_recon) ** 2, axis=1)

df["pca_error"] = recon_error

# Normalize PCA error
df["pca_score_norm"] = (
    (df["pca_error"] - df["pca_error"].min())
    / (df["pca_error"].max() - df["pca_error"].min())
)

# -----------------------------
# HYBRID RISK SCORE
# -----------------------------
IF_WEIGHT = 0.6
PCA_WEIGHT = 0.4

df["hybrid_risk_score"] = (
    IF_WEIGHT * df["if_score_norm"]
    + PCA_WEIGHT * df["pca_score_norm"]
)

# -----------------------------
# HYBRID ANOMALY FLAG
# -----------------------------
RISK_THRESHOLD = 0.6

df["hybrid_anomaly"] = (df["hybrid_risk_score"] >= RISK_THRESHOLD).astype(int)

# -----------------------------
# SAVE OUTPUT
# -----------------------------
df.to_csv("data/risk_scores/hybrid_if_pca_output.csv", index=False)

print("Hybrid IF + PCA anomaly detection complete.")
print(df["hybrid_anomaly"].value_counts())
