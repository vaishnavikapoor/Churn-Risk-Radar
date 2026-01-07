from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI(title="Customer Risk Radar API")

# Load model
model = joblib.load("churn_model.pkl")

required_cols = ['Usage Frequency', 'Payment Delay', 'Last Interaction']

# ---------- Input Schema ----------
class CustomerData(BaseModel):
    Usage_Frequency: float
    Payment_Delay: float
    Last_Interaction: float

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

    df = pd.DataFrame([[data.Usage_Frequency,
                        data.Payment_Delay,
                        data.Last_Interaction]],
                        columns=required_cols)

    prob = model.predict_proba(df)[:, 1][0]
    risk = classify_risk(prob)

    return {
        "churn_probability": float(prob),
        "risk_level": risk
    }
