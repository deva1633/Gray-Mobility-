import pandas as pd
import numpy as np

# -----------------------------
# LOAD MODEL OUTPUTS
# -----------------------------
hybrid_df = pd.read_csv("data/risk_scores/hybrid_if_pca_output.csv")
rule_df = pd.read_csv("data/risk_scores/rule_based_anomaly_output.csv")

# -----------------------------
# PARAMETERS
# -----------------------------
HIGH_RISK_THRESHOLD = 0.6      # hybrid risk
CRITICAL_RISK_THRESHOLD = 0.8
PERSIST_SECONDS = 10           # alert persistence

# -----------------------------
# INITIALIZE OUTPUT
# -----------------------------
df = hybrid_df.copy()

df["final_alert"] = 0
df["alert_level"] = "normal"
df["alert_reason"] = ""

# -----------------------------
# VALIDATION SEQUENCE
# -----------------------------
risk_buffer = []

for i in range(len(df)):

    hybrid_risk = df.loc[i, "hybrid_risk_score"]
    rule_level = rule_df.loc[i, "anomaly_level"]

    risk_buffer.append(hybrid_risk)
    if len(risk_buffer) > PERSIST_SECONDS:
        risk_buffer.pop(0)

    avg_risk = np.mean(risk_buffer)

    # -------------------------
    # DECISION LOGIC
    # -------------------------
    if avg_risk >= CRITICAL_RISK_THRESHOLD and rule_level >= 2:
        df.loc[i, "final_alert"] = 1
        df.loc[i, "alert_level"] = "CRITICAL"
        df.loc[i, "alert_reason"] = "Hybrid high risk + rule-confirmed deterioration"

    elif avg_risk >= HIGH_RISK_THRESHOLD and rule_level >= 1:
        df.loc[i, "final_alert"] = 1
        df.loc[i, "alert_level"] = "HIGH"
        df.loc[i, "alert_reason"] = "Hybrid elevated risk + early physiological confirmation"

    elif avg_risk >= HIGH_RISK_THRESHOLD and rule_level == 0:
        df.loc[i, "final_alert"] = 0
        df.loc[i, "alert_level"] = "SUPPRESSED"
        df.loc[i, "alert_reason"] = "Hybrid risk without physiological confirmation"

    else:
        df.loc[i, "final_alert"] = 0
        df.loc[i, "alert_level"] = "normal"
        df.loc[i, "alert_reason"] = "No sustained risk detected"

# -----------------------------
# SAVE OUTPUT
# -----------------------------
df.to_csv("data/risk_scores/hybrid_validated_alerts.csv", index=False)

print("Hybrid validation with rule-based gating complete.")
print(df["alert_level"].value_counts())
