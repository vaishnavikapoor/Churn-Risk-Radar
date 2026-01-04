Customer Risk Radar – Churn Prediction System

The present project takes a view on customer churn probability based on behavioral signals and categorizes customers into Low, Moderate, and High risk.

It includes:
- Logistic Regression churn model AUC 0.80
- Feature selection by correlation and importance analysis
- Batch and single-customer prediction via Streamlit web app
- Churn risk report downloadable

APPROACH
I explored the dataset and experimented with different features and models.
After multiple iterations, I found that the following three features were the most predictive:
        -Usage Frequency
        -Payment Delay
        -Last Interaction
Using these features, Logistic Regression achieved the best performance.

MODELS TRIED
        -Logistic Regression
        -Random Forest
        -XGBoost
Final Model: Logistic Regression
After iterative experimentation, identified three behavioral features that improved ROC-AUC from 0.69 to 0.806.
Best ROC-AUC Score: 0.806

RESULTS
It generates a probability of churn that is then transformed into risk levels by:
        -High Risk
        -Moderate Risk
        -Low Risk
This allows businesses to focus on strategies for retaining their customers.

TOOLS & LIBRARIES
    -Python
    -Pandas, Numpy
    -Scikit
    -matplotlib / seaborn
    -Streamlit

