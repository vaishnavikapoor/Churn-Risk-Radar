from fastapi import FastAPI
from pydantic import BaseModel
import os
import joblib

app = FastAPI(title="Customer Risk Radar API")

# Load model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "churn_model.pkl")
model = joblib.load(MODEL_PATH)


# ---------- Input Schema ----------
class CustomerData(BaseModel):
    usage_frequency: float
    payment_delay: float
    last_interaction: float

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

@app.post("/predict", response_model=PredictionOut)
def predict(data: CustomerData):

    X = [[
        data.usage_frequency,
        data.payment_delay,
        data.last_interaction
    ]]

    prob = model.predict_proba(X)[0][1]
    risk = classify_risk(prob)

    return {
        "churn_probability": float(prob),
        "risk_level": risk
    }
