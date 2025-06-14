import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the saved model and scaler
model = joblib.load('rf_final_model1.pkl')
scaler = joblib.load('scaler1.pkl')

# Define input features (replace with your actual features)
input_features = ['Credit_History_Age_Months', 'Outstanding_Debt', 'Num_Credit_Inquiries', 'Interest_Rate',
                  'Delay_from_due_date', 'Num_Bank_Accounts', 'Num_Credit_Card', 'Monthly_Balance',
                  'Annual_Income', 'Age', 'Num_of_Delayed_Payment', 'Monthly_Inhand_Salary', 'Personal_Loan',
                  'Credit_Utilization_Ratio', 'Mortgage_Loan']

st.set_page_config(page_title="Customer Category Prediction", layout="centered")
st.title("🏦 Customer Category Prediction App")

st.write("Enter customer details below to predict the category:")

# Create input form
with st.form("input_form"):
    inputs = {}
    for feature in input_features:
        inputs[feature] = st.number_input(f"{feature}", value=0.0)
    
    submitted = st.form_submit_button("Predict")

# On form submission
if submitted:
    # Convert inputs to DataFrame
    input_df = pd.DataFrame([inputs])
    
    # Scale the input data
    scaled_input = scaler.transform(input_df)
    
    # Predict using the model
    prediction = model.predict(scaled_input)[0]

    # Map prediction back to label (if needed)
    label_mapping = {
        0: "Established Customer",
        1: "Growing Customer",
        2: "Legacy Customer",
        3: "Loyal Customer",
        4: "New Customer"
    }

    st.success(f"🎯 Predicted Customer Category: **{label_mapping.get(prediction, 'Unknown')}**")
