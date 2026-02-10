from fastapi import FastAPI
from pydantic import BaseModel
import webbrowser
import threading
import time
from scripts.inference_logic import (
    compute_risk_score,
    get_risk_level,
    compute_confidence,
)

app = FastAPI(title="Gray Mobility Anomaly API")

# =========================================================
# INPUT SCHEMA
# =========================================================
class VitalsInput(BaseModel):
    heart_rate_bpm: float
    spo2_percent: float

# =========================================================
# CORE LOGIC (imported from scripts/inference_logic.py)
# =========================================================

# =========================================================
# API ENDPOINT
# =========================================================
@app.post("/predict")
def predict(vitals: VitalsInput):

    risk_score = compute_risk_score(
        vitals.heart_rate_bpm,
        vitals.spo2_percent
    )

    risk_level = get_risk_level(risk_score)
    anomaly_flag = 1 if risk_level in ["AMBER", "RED"] else 0
    confidence = compute_confidence(risk_score)

    return {
        "anomaly_flag": anomaly_flag,
        "risk_score": round(risk_score, 2),
        "risk_level": risk_level,
        "confidence": confidence
    }

# =========================================================
# AUTO-OPEN SWAGGER UI ON STARTUP
# =========================================================
def open_browser():
    time.sleep(1)  # wait for server to start
    webbrowser.open("http://127.0.0.1:8000/docs")

@app.on_event("startup")
def startup_event():
    threading.Thread(target=open_browser).start()
