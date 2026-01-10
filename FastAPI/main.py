from fastapi import FastAPI
from pydantic import BaseModel
import os
import joblib
import csv
from datetime import datetime
from typing import List

app = FastAPI(title="Customer Risk Radar API")

# Load model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "churn_model.pkl")
model = joblib.load(MODEL_PATH)

LOG_FILE = os.path.join(BASE_DIR, "logs.csv")

if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "usage_frequency", "payment_delay", "last_interaction", "churn_probability", "risk_level"])

# ---------- Input Schema ----------
class CustomerData(BaseModel):
    usage_frequency: float
    payment_delay: float
    last_interaction: float

class CustomerBatch(BaseModel):
    records: List[CustomerData]

# ---------- Output Schema ----------
class PredictionOut(BaseModel):
    churn_probability: float
    risk_level: str

# ---------- Risk Classifier ----------
def classify_risk(prob):
    if prob >= 0.7:
        return "High Risk"
    elif prob >= 0.4:
        return "Moderate Risk"
    else:
        return "Low Risk"

# ---------- Routes ----------
@app.get("/")
def home():
    return {"message": "Customer Risk Radar API is running"}

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictionOut)
def predict(data: CustomerData):

    X = [[
        data.usage_frequency,
        data.payment_delay,
        data.last_interaction
    ]]

    prob = model.predict_proba(X)[0][1]
    risk = classify_risk(prob)

    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().isoformat(),
            data.usage_frequency,
            data.payment_delay,
            data.last_interaction,
            round(float(prob), 4),
            risk
    ])

    return {
        "churn_probability": float(prob),
        "risk_level": risk
    }

@app.post("/predict-batch")
def predict_batch(data: CustomerBatch):

    X = [[
        d.usage_frequency,
        d.payment_delay,
        d.last_interaction
    ] for d in data.records]

    probs = model.predict_proba(X)[:, 1]

    results = []
    for p in probs:
        results.append({
            "churn_probability": float(p),
            "risk_level": classify_risk(p)
        })

    return {"results": results}
