import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load model and columns
model = joblib.load("churn_model.pkl")
model_columns = joblib.load("model_columns.pkl")

st.set_page_config(page_title="Customer Churn Prediction", layout="centered")

# Custom Styling
st.markdown("""
    <style>
    .main {
        background-color: #f5f7fa;
    }
    </style>
""", unsafe_allow_html=True)

st.title("📊 Customer Churn Prediction System")
st.markdown("### Predict customer churn risk using Machine Learning")

st.divider()

# Input Section
st.subheader("Enter Customer Details")

age = st.number_input("Age", 18, 100, 30)
tenure = st.number_input("Tenure (Months)", 0, 72, 5)
monthly_charges = st.number_input("Monthly Charges", 0.0, 500.0, 70.0)

gender = st.selectbox("Gender", ["Male", "Female"])
contract_type = st.selectbox("Contract Type", ["Month-to-month", "One-Year", "Two-Year"])
internet_service = st.selectbox("Internet Service", ["DSL", "Fiber Optic", "No"])
tech_support = st.selectbox("Tech Support", ["Yes", "No"])

st.divider()

if st.button("🔍 Predict Churn Risk"):

    # Prepare input
    new_customer = {
        "age": age,
        "tenure": tenure,
        "monthly_charges": monthly_charges,
        "gender": gender,
        "contract_type": contract_type,
        "internet_service": internet_service,
        "tech_support": tech_support
    }

    input_df = pd.DataFrame([new_customer])
    input_df = pd.get_dummies(input_df)

    # Align columns
    for col in model_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[model_columns]

    # Prediction
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1] * 100

    st.subheader("Prediction Result")

    # Risk indicator
    if probability < 40:
        st.success(f"🟢 Low Risk of Churn ({probability:.2f}%)")
    elif probability < 70:
        st.warning(f"🟡 Medium Risk of Churn ({probability:.2f}%)")
    else:
        st.error(f"🔴 High Risk of Churn ({probability:.2f}%)")

    st.divider()

    # Feature Importance Chart
    st.subheader("Feature Importance")

    importances = pd.Series(model.feature_importances_, index=model_columns)
    top_features = importances.sort_values(ascending=False).head(10)

    fig, ax = plt.subplots()
    top_features.plot(kind="barh", ax=ax)
    ax.set_xlabel("Importance Score")
    ax.set_ylabel("Features")
    ax.invert_yaxis()

    st.pyplot(fig)

st.divider()

st.caption("Developed by Shabaz | Customer Churn ML Project 🚀")
