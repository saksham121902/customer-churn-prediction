import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier

# =========================
# Load Trained Model
# =========================
# Before running, save your trained model in Jupyter:
#   import pickle
#   pickle.dump(xgb, open("churn_model.pkl", "wb"))

model = pickle.load(open("churn_model.pkl", "rb"))

st.title("Customer Churn Prediction App")

st.write("Enter customer details below:")

# Example input fields (customize as per dataset features)
tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12)
monthly_charges = st.number_input("Monthly Charges", min_value=0.0, max_value=200.0, value=70.0)
total_charges = st.number_input("Total Charges", min_value=0.0, max_value=10000.0, value=500.0)
contract = st.selectbox("Contract Type", ["Month-to-Month", "One year", "Two year"])
internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

# Encode categorical fields manually (simplified example)
contract_map = {"Month-to-Month":0, "One year":1, "Two year":2}
internet_map = {"DSL":0, "Fiber optic":1, "No":2}

# Create input dataframe
input_data = pd.DataFrame({
    "tenure": [tenure],
    "MonthlyCharges": [monthly_charges],
    "TotalCharges": [total_charges],
    "Contract": [contract_map[contract]],
    "InternetService": [internet_map[internet_service]]
})

# Predict churn
if st.button("Predict Churn"):
    prediction = model.predict(input_data)[0]
    if prediction == 1:
        st.error("This customer is likely to CHURN.")
    else:
        st.success("This customer is NOT likely to churn.")
