import math


def compute_risk_score(hr: float, spo2: float) -> float:
    """Compute simple risk score from HR and SpO2 for realtime inference.

    Mirrors the FastAPI logic (no model behavior changes).
    """
    risk = 0.0

    if hr > 100:
        risk += min((hr - 100) * 0.8, 30)

    if spo2 < 94:
        risk += min((94 - spo2) * 5, 50)

    return min(risk, 100.0)

def get_risk_level(risk_score: float) -> str:
    """Map risk score to triage level for realtime inference.

    Thresholds preserved from the existing API implementation.
    """
    if risk_score >= 60:
        return "RED"
    elif risk_score >= 45:
        return "AMBER"
    return "GREEN"


def compute_confidence(risk_score: float) -> float:
    """Heuristic confidence used by the API (unchanged)."""
    return round(min(0.5 + risk_score / 200.0, 0.99), 2)
