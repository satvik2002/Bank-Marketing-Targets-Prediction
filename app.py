st.subheader("📂 Or Upload a CSV File")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    csv_input_df = pd.read_csv(uploaded_file)
    
    # Scale the CSV input data
    scaled_csv_input = scaler.transform(csv_input_df)
    
    # Predict
    csv_predictions = model.predict(scaled_csv_input)
    
    label_mapping = {
        0: "Established Customer",
        1: "Growing Customer",
        2: "Legacy Customer",
        3: "Loyal Customer",
        4: "New Customer"
    }
    
    # Add predictions to dataframe
    csv_input_df['Predicted Category'] = [label_mapping.get(pred, 'Unknown') for pred in csv_predictions]
    
    st.success("✅ Prediction complete! See the results below:")
    st.dataframe(csv_input_df)
