
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Page setup
st.set_page_config(page_title="Telco Churn Predictor", layout="wide")

# 🎨 Custom CSS
st.markdown("""
<style>
/* Background image */
[data-testid="stAppViewContainer"] {
    background-image: url("https://images.unsplash.com/photo-1542223616-06776d2f9760?auto=format&fit=crop&w=1950&q=80");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}

/* Main container */
[data-testid="stAppViewContainer"] > .main {
    background-color: rgba(255, 255, 255, 0.88);
    border-radius: 15px;
    padding: 2rem;
}

/* Gradient heading */
h1 {
    font-size: 3rem !important;
    text-align: center;
    background: linear-gradient(90deg, #6a11cb 0%, #2575fc 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* Button styling */
.stButton > button {
    background-color: #2575fc;
    color: white;
    font-weight: bold;
    border-radius: 8px;
    transition: 0.3s ease;
}
.stButton > button:hover {
    background-color: #1a4db1;
}

/* Input box rounded */
.stTextInput, .stNumberInput, .stSelectbox {
    border-radius: 10px;
}

/* Hide footer */
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# 🔮 Title
st.markdown("<h1>🔮 Telco Customer Churn Prediction</h1>", unsafe_allow_html=True)

# 📊 Load and train model inside app
@st.cache_data
def load_and_train_model():
    url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
    df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(inplace=True)
    df.drop('customerID', axis=1, errors='ignore', inplace=True)
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

    le = LabelEncoder()
    for col in df.select_dtypes(include='object').columns:
        df[col] = le.fit_transform(df[col])

    X = df.drop('Churn', axis=1)
    y = df['Churn']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = RandomForestClassifier(random_state=42)
    model.fit(X_scaled, y)
    return model, scaler

# 🔁 Load model & scaler
model, scaler = load_and_train_model()

# 📥 UI Inputs
gender = st.selectbox("Gender", ["Female", "Male"])
SeniorCitizen = st.selectbox("Senior Citizen", ["No", "Yes"])
Partner = st.selectbox("Has Partner?", ["No", "Yes"])
Dependents = st.selectbox("Has Dependents?", ["No", "Yes"])
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
PaymentMethod = st.selectbox("Payment Method", [
    "Electronic check", "Mailed check",
    "Bank transfer (automatic)", "Credit card (automatic)"
])
MonthlyCharges = st.number_input("Monthly Charges", min_value=0.0, max_value=200.0, value=70.0)
TotalCharges = st.number_input("Total Charges", min_value=0.0, max_value=10000.0, value=2000.0)

# Helper to encode
def encode(val, mapping):
    return mapping[val]

# 🔮 Prediction
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
