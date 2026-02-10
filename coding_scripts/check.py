import pandas as pd

df = pd.read_csv("data/risk_scores/final_decision.csv")

print(df["risk_score"].describe())
print(df["risk_level"].value_counts(normalize=True) * 100)
print(
    df.groupby("risk_level")[["heart_rate_bpm", "spo2_percent"]].mean()
)