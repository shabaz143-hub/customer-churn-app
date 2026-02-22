import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score

# Load dataset
df = pd.read_csv("data2.csv")

print("Dataset Loaded Successfully!")
print("Shape:", df.shape)

# Clean column names
df.columns = df.columns.str.lower().str.strip()

# Drop customer_id
if "customer_id" in df.columns:
    df = df.drop("customer_id", axis=1)

# Remove total_charges (VERY IMPORTANT – causes leakage)
if "total_charges" in df.columns:
    df = df.drop("total_charges", axis=1)

# Target column
churn_column = "churn_status"

print("\nChurn Distribution:")
print(df[churn_column].value_counts())

# Convert Yes/No if needed
if df[churn_column].dtype == "object":
    df[churn_column] = df[churn_column].map({"Yes":1, "No":0})

# Drop missing
df = df.dropna()

# Convert categorical variables
df = pd.get_dummies(df, drop_first=True)

# Split X and y
X = df.drop(churn_column, axis=1)
print("\nTraining Columns:")
print(X.columns)

y = df[churn_column]

# Stratified split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=1,
    stratify=y
)

# Build more realistic model
model = RandomForestClassifier(
    n_estimators=50,
    max_depth=4,
    min_samples_split=10,
    class_weight="balanced",
    random_state=1
)

model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:,1]

# Evaluation
print("\nAccuracy:", round(accuracy_score(y_test, y_pred)*100,2), "%")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("ROC AUC Score:", round(roc_auc_score(y_test, y_prob),4))

import joblib

# Save training columns
joblib.dump(X.columns.tolist(), "model_columns.pkl")

# Save model
joblib.dump(model, "churn_model.pkl")

print("\nModel and columns saved successfully!")
