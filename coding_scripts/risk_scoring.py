import pandas as pd
import numpy as np
import os

# -------------------------
# FILE PATHS
# -------------------------
INPUT_FILE = "data/risk_scores/hybrid_validated_alerts.csv"
OUTPUT_FILE = "data/risk_scores/final_decision.csv"

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(
        f"{INPUT_FILE} not found.\nFiles in data/risk_scores/: {os.listdir('data/risk_scores') if os.path.exists('data/risk_scores') else 'missing'}"
    )

# -------------------------
# LOAD DATA
# -------------------------
df = pd.read_csv(INPUT_FILE)

print("✅ Loaded hybrid validated alerts")
print("Columns:", df.columns.tolist())

# -------------------------
# DEFINE CORRECT COLUMNS
# -------------------------
anomaly_col = "hybrid_risk_score"     # combined IF + PCA score
rule_col = "hybrid_anomaly"           # validated anomaly flag (0/1)

# -------------------------
# TREND SEVERITY (early warning)
# -------------------------
# Use meaningful trend features (already engineered by you)
trend_features = [
    "hr_slope_30s",
    "spo2_slope_60s",
    "sys_bp_slope_60s"
]

# Keep only features that exist
trend_features = [c for c in trend_features if c in df.columns]

# Aggregate trend severity
df["trend_severity"] = df[trend_features].abs().mean(axis=1)

# -------------------------
# NORMALIZATION (0–100)
# -------------------------
def normalize(series):
    return 100 * (series - series.min()) / (series.max() - series.min() + 1e-6)

df["anomaly_norm"] = normalize(df[anomaly_col])
df["trend_norm"] = normalize(df["trend_severity"])

# Confidence proxy:
# stronger anomaly + consistent trend → higher confidence
df["confidence"] = (
    0.6 * df["anomaly_norm"] + 0.4 * df["trend_norm"]
) / 100

# -------------------------
# FINAL RISK SCORE (Task 2B)
# -------------------------
df["risk_score"] = (
    0.6 * df["anomaly_norm"] +
    0.3 * df["trend_norm"] +
    0.1 * (df["confidence"] * 100)
)

# -------------------------
# TRIAGE LEVELS
# -------------------------
def risk_level(score):
    if score >= 70:
        return "RED"
    elif score >= 40:
        return "AMBER"
    return "GREEN"

df["risk_level"] = df["risk_score"].apply(risk_level)

# -------------------------
# FINAL ALERT DECISION
# -------------------------
def final_alert(row):
    if row[rule_col] == 0:
        return 0
    return 1 if row["risk_level"] == "RED" else 0

df["final_alert_flag"] = df.apply(final_alert, axis=1)

# -------------------------
# SAVE OUTPUT
# -------------------------
df.to_csv(OUTPUT_FILE, index=False)

print("✅ Risk scoring completed successfully")
print(df["risk_level"].value_counts())
print(f"📁 Output saved to {OUTPUT_FILE}")
