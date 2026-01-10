# Customer Risk Radar – Churn Prediction System

Live App:
https://churn-risk-radar-e5c7fe6v8d64nuuo2q2adh.streamlit.app/

Customer Risk Radar predicts customer churn probability using behavioral signals and categorizes customers into Low, Moderate, and High risk.

------------------------------------------------------------

PROBLEM OVERVIEW

Customer churn has a direct impact on revenue. This system identifies churn risk using three key behavioral features.

Feature | Description
Usage Frequency | Number of platform interactions in last 30 days
Payment Delay | Total delayed payment days in last billing cycle
Last Interaction | Days since last customer activity

------------------------------------------------------------

MODELING APPROACH

Multiple models were explored using correlation analysis and feature importance.

Models tried:
- Logistic Regression
- Random Forest
- XGBoost

After iterative experimentation, the following three features were selected:
- Usage Frequency
- Payment Delay
- Last Interaction

These features improved ROC-AUC from 0.69 to 0.806.

Final Model: Logistic Regression  
Best ROC-AUC: 0.806

------------------------------------------------------------

INPUT FORMAT

CSV file must contain the following columns:

Column Name | Type | Range
Usage Frequency | int | 0 – 100
Payment Delay | int | 0 – 60
Last Interaction | int | 0 – 90

Customer IDs may be included.

------------------------------------------------------------

APPLICATION FEATURES

- Single-customer churn prediction
- Batch CSV churn prediction
- Downloadable churn risk report
- Risk categorization: Low, Moderate, High
- End-to-end deployed ML system

------------------------------------------------------------

TECH STACK

- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn
- Streamlit
- FastAPI

------------------------------------------------------------

RESULT

The system generates churn probabilities and assigns actionable risk levels to support customer retention strategies.

------------------------------------------------------------

WHY THIS PROJECT MATTERS

This project demonstrates a full machine learning deployment lifecycle including feature engineering, model optimization, batch inference pipelines, and production-ready web integration.
