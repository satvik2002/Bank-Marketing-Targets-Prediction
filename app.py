import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model and scaler
model = joblib.load('rf_final_model.pkl')
scaler = joblib.load('scaler.pkl')

# Define input features
input_features = [
    'Credit_History_Age_Months', 'Outstanding_Debt', 'Num_Credit_Inquiries', 'Interest_Rate',
    'Delay_from_due_date', 'Num_Bank_Accounts', 'Num_Credit_Card', 'Monthly_Balance',
    'Annual_Income', 'Age', 'Num_of_Delayed_Payment', 'Monthly_Inhand_Salary',
    'Personal_Loan', 'Credit_Utilization_Ratio', 'Mortgage_Loan'
]

# Label mapping
label_mapping = {
    0: "Established Customer",
    1: "Growing Customer",
    2: "Legacy Customer",
    3: "Loyal Customer",
    4: "New Customer"
}

# Page config
st.set_page_config(page_title="Customer Category Prediction", layout="centered")
st.title("🏦 Customer Category Prediction App")

st.write("Enter customer details below to predict the category or upload a CSV file.")

# --------- Section 1: Manual Input ---------
st.subheader("📋 Manual Input")

with st.form("input_form"):
    inputs = {}
    for feature in input_features:
        inputs[feature] = st.number_input(f"{feature}", value=0.0)

    submitted = st.form_submit_button("Predict")

if submitted:
    input_df = pd.DataFrame([inputs])
    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)[0]
    st.success(f"🎯 Predicted Customer Category: **{label_mapping.get(prediction, 'Unknown')}**")

# --------- Section 2: CSV Upload ---------
st.subheader("📂 Or Upload a CSV File")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    try:
        csv_input_df = pd.read_csv(uploaded_file)

        # Check if all required features are present
        missing_cols = set(input_features) - set(csv_input_df.columns)
        if missing_cols:
            st.error(f"❌ Missing columns in uploaded CSV: {', '.join(missing_cols)}")
        else:
            scaled_csv_input = scaler.transform(csv_input_df[input_features])
            csv_predictions = model.predict(scaled_csv_input)
            csv_input_df['Predicted Category'] = [label_mapping.get(pred, 'Unknown') for pred in csv_predictions]
            st.success("✅ Prediction complete! See results below:")
            st.dataframe(csv_input_df)
    except Exception as e:
        st.error(f"❌ Error processing file: {e}")
