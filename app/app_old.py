import streamlit as st
import pandas as pd
import joblib

# Load saved files
model = joblib.load("../models/churn_model.pkl")
scaler = joblib.load("../models/scaler.pkl")
model_features = joblib.load("../models/model_features.pkl")

st.title("Customer Churn Prediction System")

st.write("Enter customer details below:")

# User Inputs
tenure = st.slider("Tenure (months)", 0, 72, 12)

monthly_charges = st.number_input(
    "Monthly Charges",
    min_value=0.0,
    value=50.0
)

total_charges = st.number_input(
    "Total Charges",
    min_value=0.0,
    value=500.0
)

contract = st.selectbox(
    "Contract",
    ["Month-to-month", "One year", "Two year"]
)

gender = st.selectbox(
    "Gender",
    ["Female","Male"]
)

# Create input dictionary
input_data = {
    "tenure": tenure,
    "MonthlyCharges": monthly_charges,
    "TotalCharges": total_charges
}

# Convert to DataFrame
input_df = pd.DataFrame([input_data])

# Add missing columns
for col in model_features:
    if col not in input_df.columns:
        input_df[col] = 0

input_df = input_df[model_features]

# Scale input
input_scaled = scaler.transform(input_df)

# Predict
if st.button("Predict Churn"):

    prediction = model.predict(
        input_scaled
    )[0]

    probability = model.predict_proba(
        input_scaled
    )[0][1]

    if prediction == 1:
        st.error(
            f"High Churn Risk ({probability:.2%})"
        )

    else:
        st.success(
            f"Low Churn Risk ({probability:.2%})"
        )