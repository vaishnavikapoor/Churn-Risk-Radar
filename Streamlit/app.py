import streamlit as st
import pandas as pd
import requests
import time

# Force no HTTP caching
requests.sessions.Session.trust_env = False

API_URL = "https://churn-risk-radar.onrender.com"

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

        try:
            res = requests.post(
                f"{API_URL}/predict",
                json=payload,
                timeout=10
            )


            if res.status_code != 200:
                st.error(f"Backend error: {res.text}")
            else:
                response = res.json()
                st.success(f"Risk Level: {response['risk_level']}")
                st.metric("Churn Probability", f"{response['churn_probability']*100:.2f}%")

        except Exception as e:
            st.error(f"Connection failed: {e}")

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

            payload = {
                "records": [
                    {
                        "usage_frequency": float(row["Usage Frequency"]),
                        "payment_delay": float(row["Payment Delay"]),
                        "last_interaction": float(row["Last Interaction"])
                    }       
                    for _, row in data.iterrows()
                ]
            }

            with st.spinner("Processing batch..."):
                res = requests.post(
                    f"{API_URL}/predict-batch",
                    json=payload,
                    timeout=30
                )

            if res.status_code != 200:
                st.error(f"Backend error: {res.text}")
            else:
                results = res.json()["results"]

                data["Churn Probability"] = [r["churn_probability"] for r in results]
                data["Risk Level"] = [r["risk_level"] for r in results]

                st.dataframe(data.head())
                st.download_button(
                    "Download Risk Report",
                    data.to_csv(index=False),
                    "churn_risk_report.csv",
                    "text/csv"
                )

            data["Churn Probability"] = [r[0] for r in results]
            data["Risk Level"] = [r[1] for r in results]

            st.dataframe(data.head())
            st.download_button(
                "Download Risk Report",
                data.to_csv(index=False),
                "churn_risk_report.csv",
                "text/csv"
            )
