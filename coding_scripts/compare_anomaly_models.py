import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

# -----------------------------
# LOAD OUTPUTS
# -----------------------------
rule_df = pd.read_csv("data/risk_scores/rule_based_anomaly_output.csv")
if_df = pd.read_csv("data/risk_scores/isolation_forest_output.csv")
pca_df = pd.read_csv("data/risk_scores/pca_anomaly_output.csv")

# -----------------------------
# DEFINE GROUND TRUTH
# Distress = 10–20 minutes
# -----------------------------
time_min = rule_df["time_sec"] / 60
ground_truth = ((time_min >= 10) & (time_min <= 20)).astype(int)

# -----------------------------
# EXTRACT MODEL PREDICTIONS
# -----------------------------
rule_pred = (rule_df["anomaly_level"] >= 2).astype(int)
if_pred = if_df["if_anomaly"]
pca_pred = pca_df["pca_anomaly"]

# -----------------------------
# METRIC FUNCTION
# -----------------------------
def compute_metrics(y_true, y_pred):
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    false_alert_rate = (
        ((y_pred == 1) & (y_true == 0)).sum()
        / max((y_true == 0).sum(), 1)
    )

    return precision, recall, f1, false_alert_rate

# -----------------------------
# LATENCY CALCULATION
# -----------------------------
def detection_latency(y_true, y_pred, time_sec):
    true_start = time_sec[y_true == 1].min()
    detected = time_sec[(y_pred == 1) & (y_true == 1)]

    if len(detected) == 0:
        return np.inf

    return detected.min() - true_start

# -----------------------------
# COMPUTE METRICS
# -----------------------------
results = []

for name, pred in [
    ("Rule-based", rule_pred),
    ("Isolation Forest", if_pred),
    ("PCA", pca_pred),
]:
    precision, recall, f1, far = compute_metrics(ground_truth, pred)
    latency = detection_latency(
        ground_truth.values,
        pred.values,
        rule_df["time_sec"].values
    )

    results.append({
        "Model": name,
        "Precision": round(precision, 3),
        "Recall": round(recall, 3),
        "F1-score": round(f1, 3),
        "False Alert Rate": round(far, 3),
        "Detection Latency (sec)": int(latency) if latency != np.inf else "Not detected"
    })

# -----------------------------
# DISPLAY RESULTS
# -----------------------------
results_df = pd.DataFrame(results)
print("\n=== Model Comparison Metrics ===\n")
print(results_df)

results_df.to_csv("analysis/metrics/model_comparison_metrics.csv", index=False)
