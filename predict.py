import pandas as pd
import joblib

# Load model and training columns
model = joblib.load("churn_model.pkl")
model_columns = joblib.load("model_columns.pkl")

# Raw customer input (ORIGINAL VALUES)
new_customer = {
    "age": 30,
    "tenure": 5,
    "monthly_charges": 70,
    "gender": "Male",
    "contract_type": "Month-to-month",
    "internet_service": "Fiber Optic",
    "tech_support": "No"
}

# Convert to DataFrame
input_df = pd.DataFrame([new_customer])

# Apply same preprocessing
input_df = pd.get_dummies(input_df)

# Add missing columns
for col in model_columns:
    if col not in input_df.columns:
        input_df[col] = 0

# Keep correct column order
input_df = input_df[model_columns]

# Predict
prediction = model.predict(input_df)

if prediction[0] == 1:
    print("Customer is likely to Churn")
else:
    print("Customer is NOT likely to Churn")
