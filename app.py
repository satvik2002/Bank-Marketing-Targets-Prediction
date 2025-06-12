import streamlit as st
import pandas as pd
import joblib
import os

# --- Load Pickle Files ---
def load_file(path, name):
    if not os.path.exists(path):
        st.error(f"❌ {name} file not found.")
        st.stop()
    return joblib.load(path)

model = load_file('rf_final_model1.pkl', 'Model')
scaler = load_file('scaler1.pkl', 'Scaler')
label_encoder = load_file('LabelEncoder1.pkl', 'Label Encoder')

# --- Feature Inputs ---
input_features = [
    'Credit_History_Age_Months', 'Outstanding_Debt', 'Num_Credit_Inquiries', 'Interest_Rate',
    'Delay_from_due_date', 'Num_Bank_Accounts', 'Num_Credit_Card', 'Monthly_Balance',
    'Annual_Income', 'Age', 'Num_of_Delayed_Payment', 'Monthly_Inhand_Salary',
    'Personal_Loan', 'Credit_Utilization_Ratio', 'Mortgage_Loan'
]

# --- Page Title ---
st.set_page_config(page_title="Customer Category Prediction", layout="centered")
st.title("🏦 Customer Category Prediction")

# --- Choose Mode ---
mode = st.radio("Choose Input Method:", ["🔘 Manual Entry", "📁 CSV Upload"], horizontal=True)

# --- Manual Entry ---
if mode == "🔘 Manual Entry":
    st.subheader("📋 Enter Customer Details")
    with st.form("manual_input"):
        inputs = {feature: st.number_input(feature, value=0.0) for feature in input_features}
        submit = st.form_submit_button("Predict")
    
    if submit:
        input_df = pd.DataFrame([inputs])
        scaled = scaler.transform(input_df)
        pred_encoded = model.predict(scaled)[0]
        pred_label = label_encoder.inverse_transform([pred_encoded])[0]

        st.success(f"🎯 Predicted Category: **{pred_label}**")

# --- CSV Upload ---
else:
    st.subheader("📁 Upload CSV File")
    uploaded_file = st.file_uploader("Upload your CSV file", type="csv")

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        missing = set(input_features) - set(df.columns)
        if missing:
            st.error(f"Missing columns in uploaded CSV: {', '.join(missing)}")
        else:
            scaled = scaler.transform(df[input_features])
            predictions = model.predict(scaled)
            df['Predicted_Category'] = label_encoder.inverse_transform(predictions)
            st.success("✅ Prediction complete.")
            st.dataframe(df)
