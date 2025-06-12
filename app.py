import streamlit as st
import pandas as pd
import joblib

# Load model, scaler, and label encoder
model = joblib.load('rf_final_model1.pkl')
scaler = joblib.load('scaler1.pkl')
label_encoder = joblib.load('LabelEncoder1.pkl')  # Load LabelEncoder

# Define features used for input
input_features = [
    'Credit_History_Age_Months', 'Outstanding_Debt', 'Num_Credit_Inquiries', 'Interest_Rate',
    'Delay_from_due_date', 'Num_Bank_Accounts', 'Num_Credit_Card', 'Monthly_Balance',
    'Annual_Income', 'Age', 'Num_of_Delayed_Payment', 'Monthly_Inhand_Salary',
    'Personal_Loan', 'Credit_Utilization_Ratio', 'Mortgage_Loan'
]

# --- Initialize session state ---
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'username' not in st.session_state:
    st.session_state.username = ''

# --- Login Function ---
def login():
    st.title("🔐 Login Page")
    st.subheader("Please enter your credentials:")

    username = st.text_input("Username")
    password = st.text_input("Password", type='password')
    if st.button("Login"):
        if username == "admin" and password == "admin123":
            st.session_state.logged_in = True
            st.session_state.username = username
            st.success("✅ Login successful!")
            st.rerun()
        else:
            st.error("❌ Invalid credentials. Try again.")

# --- Logout Function ---
def logout():
    st.session_state.logged_in = False
    st.session_state.username = ''
    st.success("✅ Logged out successfully.")
    st.rerun()

# --- Main App after login ---
def main_app():
    st.set_page_config(page_title="Customer Category Prediction", layout="centered")
    st.title("🏦 Customer Category Prediction App")

    # Page switcher
    page = st.radio("Choose Page:", ["🔘 Manual Input", "📂 CSV Upload"], horizontal=True)

    if st.button("Logout"):
        logout()

    # --- Manual Input ---
    if page == "🔘 Manual Input":
        st.subheader("📋 Manual Customer Entry")
        with st.form("manual_form"):
            inputs = {feature: st.number_input(f"{feature}", value=0.0) for feature in input_features}
            submitted = st.form_submit_button("Predict")
        if submitted:
            input_df = pd.DataFrame([inputs])
            scaled_input = scaler.transform(input_df)
            prediction = model.predict(scaled_input)[0]

            # Decode label using label encoder
            pred_label = label_encoder.inverse_transform([prediction])[0]

            st.success(f"🎯 Predicted Customer Category: **{pred_label}**")

    # --- CSV Upload ---
    elif page == "📂 CSV Upload":
        st.subheader("📁 Upload CSV for Bulk Prediction")
        uploaded_file = st.file_uploader("Upload CSV", type='csv')
        if uploaded_file is not None:
            try:
                csv_df = pd.read_csv(uploaded_file)
                missing = set(input_features) - set(csv_df.columns)
                if missing:
                    st.error(f"Missing columns: {', '.join(missing)}")
                else:
                    scaled = scaler.transform(csv_df[input_features])
                    predictions = model.predict(scaled)
                    decoded_predictions = label_encoder.inverse_transform(predictions)
                    csv_df['Predicted Category'] = decoded_predictions
                    st.success("✅ Prediction complete")
                    st.dataframe(csv_df)
            except Exception as e:
                st.error(f"Error: {e}")

# --- Run App ---
if not st.session_state.logged_in:
    login()
else:
    main_app()
