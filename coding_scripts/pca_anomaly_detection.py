import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# -----------------------------
# LOAD FEATURE DATA
# -----------------------------
df = pd.read_csv("data/processed/features.csv")

# -----------------------------
# SELECT FEATURES
# (same ones used for Isolation Forest)
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
# FIT PCA
# -----------------------------
# Keep enough components to explain most variance
pca = PCA(n_components=0.95, random_state=42)
X_pca = pca.fit_transform(X_scaled)

# -----------------------------
# RECONSTRUCTION ERROR
# -----------------------------
X_reconstructed = pca.inverse_transform(X_pca)
reconstruction_error = np.mean(
    (X_scaled - X_reconstructed) ** 2, axis=1
)

df["pca_reconstruction_error"] = reconstruction_error

# -----------------------------
# ANOMALY THRESHOLD
# -----------------------------
# Use high percentile (unsupervised)
THRESHOLD_PERCENTILE = 95
threshold = np.percentile(reconstruction_error, THRESHOLD_PERCENTILE)

df["pca_anomaly"] = (reconstruction_error > threshold).astype(int)

# -----------------------------
# NORMALIZE SCORE (0–1)
# -----------------------------
df["pca_score_norm"] = (
    (df["pca_reconstruction_error"] - df["pca_reconstruction_error"].min())
    / (df["pca_reconstruction_error"].max() - df["pca_reconstruction_error"].min())
)

# -----------------------------
# SAVE OUTPUT
# -----------------------------
df.to_csv("data/risk_scores/pca_anomaly_output.csv", index=False)

print("PCA anomaly detection complete.")
print("Explained variance ratio:", pca.explained_variance_ratio_)
print(df["pca_anomaly"].value_counts())
