import numpy as np
import pandas as pd

# -----------------------------
# CONFIG
# -----------------------------
SEED = 42
DURATION_MINUTES = 30
SAMPLING_RATE_HZ = 1  # 1 value per second
TOTAL_SECONDS = DURATION_MINUTES * 60

np.random.seed(SEED)

# -----------------------------
# TIME INDEX
# -----------------------------
time = np.arange(TOTAL_SECONDS)

# -----------------------------
# PHASE DEFINITIONS (in seconds)
# -----------------------------
NORMAL_START = 0
NORMAL_END = 10 * 60          # first 10 minutes
DISTRESS_START = 10 * 60
DISTRESS_END = 20 * 60        # next 10 minutes
RECOVERY_START = 20 * 60
RECOVERY_END = TOTAL_SECONDS # last 10 minutes

# -----------------------------
# MOTION SIGNAL (vehicle + patient)
# -----------------------------
motion = np.random.normal(0.2, 0.05, TOTAL_SECONDS)

# vehicle bumps
for t in range(300, TOTAL_SECONDS, 180):
    motion[t:t+5] += np.random.uniform(0.6, 1.0)

# distress-related patient movement
motion[DISTRESS_START:DISTRESS_END] += np.random.normal(0.3, 0.1, DISTRESS_END - DISTRESS_START)

motion = np.clip(motion, 0, None)

# -----------------------------
# HEART RATE (bpm)
# -----------------------------
hr = np.zeros(TOTAL_SECONDS)

# normal
hr[NORMAL_START:NORMAL_END] = np.random.normal(75, 3, NORMAL_END - NORMAL_START)

# distress (gradual rise)
hr[DISTRESS_START:DISTRESS_END] = np.linspace(85, 120, DISTRESS_END - DISTRESS_START) \
                                  + np.random.normal(0, 4, DISTRESS_END - DISTRESS_START)

# recovery
hr[RECOVERY_START:RECOVERY_END] = np.linspace(100, 80, RECOVERY_END - RECOVERY_START) \
                                  + np.random.normal(0, 3, RECOVERY_END - RECOVERY_START)

# motion-induced HR spikes
hr += motion * np.random.uniform(5, 10)

# -----------------------------
# SpO2 (%)
# -----------------------------
spo2 = np.zeros(TOTAL_SECONDS)

# normal
spo2[NORMAL_START:NORMAL_END] = np.random.normal(98, 0.5, NORMAL_END - NORMAL_START)

# distress (drop)
spo2[DISTRESS_START:DISTRESS_END] = np.linspace(96, 90, DISTRESS_END - DISTRESS_START) \
                                    + np.random.normal(0, 0.7, DISTRESS_END - DISTRESS_START)

# recovery
spo2[RECOVERY_START:RECOVERY_END] = np.linspace(92, 97, RECOVERY_END - RECOVERY_START) \
                                    + np.random.normal(0, 0.5, RECOVERY_END - RECOVERY_START)

# motion-induced SpO2 artifacts (false drops)
artifact_mask = motion > 0.7
spo2[artifact_mask] -= np.random.uniform(3, 8)

spo2 = np.clip(spo2, 75, 100)

# -----------------------------
# BLOOD PRESSURE (mmHg)
# -----------------------------
sys_bp = np.zeros(TOTAL_SECONDS)
dia_bp = np.zeros(TOTAL_SECONDS)

# normal
sys_bp[NORMAL_START:NORMAL_END] = np.random.normal(120, 5, NORMAL_END - NORMAL_START)
dia_bp[NORMAL_START:NORMAL_END] = np.random.normal(80, 4, NORMAL_END - NORMAL_START)

# distress
sys_bp[DISTRESS_START:DISTRESS_END] = np.linspace(125, 95, DISTRESS_END - DISTRESS_START) \
                                      + np.random.normal(0, 6, DISTRESS_END - DISTRESS_START)
dia_bp[DISTRESS_START:DISTRESS_END] = np.linspace(85, 60, DISTRESS_END - DISTRESS_START) \
                                      + np.random.normal(0, 5, DISTRESS_END - DISTRESS_START)

# recovery
sys_bp[RECOVERY_START:RECOVERY_END] = np.linspace(100, 118, RECOVERY_END - RECOVERY_START) \
                                      + np.random.normal(0, 4, RECOVERY_END - RECOVERY_START)
dia_bp[RECOVERY_START:RECOVERY_END] = np.linspace(65, 78, RECOVERY_END - RECOVERY_START) \
                                      + np.random.normal(0, 3, RECOVERY_END - RECOVERY_START)

# -----------------------------
# MISSING DATA (sensor dropout)
# -----------------------------
dropout_indices = np.random.choice(TOTAL_SECONDS, size=40, replace=False)

for idx in dropout_indices:
    hr[idx] = np.nan
    spo2[idx] = np.nan
    sys_bp[idx] = np.nan
    dia_bp[idx] = np.nan

# -----------------------------
# FINAL DATAFRAME
# -----------------------------
df = pd.DataFrame({
    "time_sec": time,
    "heart_rate_bpm": hr,
    "spo2_percent": spo2,
    "bp_systolic": sys_bp,
    "bp_diastolic": dia_bp,
    "motion": motion
})

# -----------------------------
# SAVE
# -----------------------------
df.to_csv("synthetic_ambulance_vitals.csv", index=False)

print("Synthetic ambulance vitals dataset generated:")
print(df.head())
