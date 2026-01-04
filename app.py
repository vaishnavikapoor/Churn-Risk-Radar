# Streamlit app for batch churn prediction
# Accepts CSV uploads and returns risk-segmented output file

import streamlit as st
import pandas as pd
import joblib

model = joblib.load("churn_model.pkl")

st.set_page_config(page_title="Customer Risk Radar", layout="centered")

st.title("Customer Risk Radar")
st.write("Predict customer churn risk using behavior signals.")

required_cols = ['Usage Frequency', 'Payment Delay', 'Last Interaction']

def classify_risk(prob):
    if prob > 0.7:
        return "High Risk"
    elif prob > 0.4:
        return "Moderate"
    else:
        return "Safe"

mode = st.radio("Select Mode", ["Single Customer", "Bulk CSV Upload"])

# ---------------- SINGLE CUSTOMER MODE ----------------
if mode == "Single Customer":
    st.subheader("Single Customer Prediction")

    usage = st.slider("Usage Frequency", 0, 100, 50)
    delay = st.slider("Payment Delay (days)", 0, 30, 5)
    last_interaction = st.slider("Days Since Last Interaction", 0, 60, 10)

    input_df = pd.DataFrame([[usage, delay, last_interaction]],
                            columns=required_cols)

    prob = model.predict_proba(input_df)[:,1][0]
    risk = classify_risk(prob)

    st.metric("Churn Probability", f"{prob:.2f}")
    st.success(f"Risk Level: {risk}")

# ---------------- BULK MODE ----------------
else:
    st.subheader("Bulk Customer Risk Prediction")

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file:
        data = pd.read_csv(uploaded_file)

        if not all(col in data.columns for col in required_cols):
            st.error("CSV must contain: Usage Frequency, Payment Delay, Last Interaction")
        else:
            X = data[required_cols]
            probs = model.predict_proba(X)[:,1]

            result = pd.DataFrame()
            result["CustomerID"] = data.index
            result["Churn Probability"] = probs
            result["Risk Level"] = result["Churn Probability"].apply(classify_risk)

            st.dataframe(result.head())

            csv = result.to_csv(index=False, encoding="utf-8-sig")
            st.download_button("Download Risk Report", csv, "churn_risk_report.csv", "text/csv")
