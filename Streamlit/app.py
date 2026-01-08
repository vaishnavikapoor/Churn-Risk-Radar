# Streamlit app for batch churn prediction using FastAPI backend

import streamlit as st
import pandas as pd
import requests

API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(page_title="Customer Risk Radar", layout="centered")
st.title("Customer Risk Radar")
st.write("Predict customer churn risk using behavior signals.")

required_cols = ['Usage Frequency', 'Payment Delay', 'Last Interaction']

mode = st.radio("Select Mode", ["Single Customer", "Bulk CSV Upload"])

# ---------------- SINGLE CUSTOMER MODE ----------------
if mode == "Single Customer":
    st.subheader("Single Customer Prediction")

    usage = st.slider("Usage Frequency", 0, 100, 50)
    delay = st.slider("Payment Delay (days)", 0, 30, 5)
    last_interaction = st.slider("Days Since Last Interaction", 0, 60, 10)

    if st.button("Predict Risk"):
        payload = {
            "usage_frequency": usage,
            "payment_delay": delay,
            "last_interaction": last_interaction
        }

        res = requests.post(API_URL, json=payload)

        if res.status_code != 200:
            st.error(f"Backend error: {res.text}")
        else:
            response = res.json()
            st.success(f"Risk Level: {response['risk_level']}")
            st.metric("Churn Probability", f"{response['churn_probability']*100:.1f}%")


# ---------------- BULK MODE ----------------
else:
    st.subheader("Bulk Customer Risk Prediction")

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file:
        data = pd.read_csv(uploaded_file)

        if not all(col in data.columns for col in required_cols):
            st.error("CSV must contain: Usage Frequency, Payment Delay, Last Interaction")
        else:
            results = []

            for _, row in data.iterrows():
                payload = {
                    "usage_frequency": row["Usage Frequency"],
                    "payment_delay": row["Payment Delay"],
                    "last_interaction": row["Last Interaction"]
                }

                res = requests.post(API_URL, json=payload).json()

                results.append([
                    res["churn_probability"],
                    res["risk_level"]
                ])

            data["Churn Probability"] = [r[0] for r in results]
            data["Risk Level"] = [r[1] for r in results]

            st.dataframe(data.head())

            csv = data.to_csv(index=False, encoding="utf-8-sig")
            st.download_button("Download Risk Report", csv, "churn_risk_report.csv", "text/csv")

