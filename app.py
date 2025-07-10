import streamlit as st
import pandas as pd
import joblib

# Load model and scaler
model = joblib.load("best_churn_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("🔮 Telco Customer Churn Prediction")

# Input fields
gender = st.selectbox("Gender", ["Female", "Male"])
SeniorCitizen = st.selectbox("Senior Citizen", ["No", "Yes"])
Partner = st.selectbox("Partner", ["No", "Yes"])
Dependents = st.selectbox("Dependents", ["No", "Yes"])
tenure = st.slider("Tenure (months)", 0, 72, 12)
PhoneService = st.selectbox("Phone Service", ["No", "Yes"])
MultipleLines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
OnlineSecurity = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
OnlineBackup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
DeviceProtection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
TechSupport = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
StreamingTV = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
StreamingMovies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
PaperlessBilling = st.selectbox("Paperless Billing", ["No", "Yes"])
PaymentMethod = st.selectbox("Payment Method", ["Electronic check", "Mailed check", 
    "Bank transfer (automatic)", "Credit card (automatic)"])
MonthlyCharges = st.number_input("Monthly Charges", min_value=0.0, max_value=200.0, value=70.0)
TotalCharges = st.number_input("Total Charges", min_value=0.0, max_value=10000.0, value=2000.0)

def encode(val, mapping):
    return mapping[val]

if st.button("Predict Churn"):
    data = {
        'gender': encode(gender, {"Female": 0, "Male": 1}),
        'SeniorCitizen': encode(SeniorCitizen, {"No": 0, "Yes": 1}),
        'Partner': encode(Partner, {"No": 0, "Yes": 1}),
        'Dependents': encode(Dependents, {"No": 0, "Yes": 1}),
        'tenure': tenure,
        'PhoneService': encode(PhoneService, {"No": 0, "Yes": 1}),
        'MultipleLines': encode(MultipleLines, {"No": 0, "Yes": 1, "No phone service": 2}),
        'InternetService': encode(InternetService, {"DSL": 0, "Fiber optic": 1, "No": 2}),
        'OnlineSecurity': encode(OnlineSecurity, {"No": 0, "Yes": 1, "No internet service": 2}),
        'OnlineBackup': encode(OnlineBackup, {"No": 0, "Yes": 1, "No internet service": 2}),
        'DeviceProtection': encode(DeviceProtection, {"No": 0, "Yes": 1, "No internet service": 2}),
        'TechSupport': encode(TechSupport, {"No": 0, "Yes": 1, "No internet service": 2}),
        'StreamingTV': encode(StreamingTV, {"No": 0, "Yes": 1, "No internet service": 2}),
        'StreamingMovies': encode(StreamingMovies, {"No": 0, "Yes": 1, "No internet service": 2}),
        'Contract': encode(Contract, {"Month-to-month": 0, "One year": 1, "Two year": 2}),
        'PaperlessBilling': encode(PaperlessBilling, {"No": 0, "Yes": 1}),
        'PaymentMethod': encode(PaymentMethod, {
            "Electronic check": 0, "Mailed check": 1,
            "Bank transfer (automatic)": 2, "Credit card (automatic)": 3}),
        'MonthlyCharges': MonthlyCharges,
        'TotalCharges': TotalCharges
    }

    input_df = pd.DataFrame([data])
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)

    if prediction[0] == 1:
        st.error("⚠️ The customer is likely to churn.")
    else:
        st.success("✅ The customer is likely to stay.")
