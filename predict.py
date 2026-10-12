import joblib
import pandas as pd

# Load the trained model
model = joblib.load("churn_model.pkl")

# Get customer details
tenure = int(input("Customer tenure (months, 0-72): "))
contract = input(
    "Contract (Month-to-month, One year, Two year): "
).strip()
monthly_charges = float(input("Monthly charges: "))

# Validate inputs
if not 0 <= tenure <= 72:
    raise ValueError("Tenure must be between 0 and 72 months.")

valid_contracts = ["Month-to-month", "One year", "Two year"]

if contract not in valid_contracts:
    raise ValueError(
        "Invalid contract. Choose Month-to-month, One year, or Two year."
    )

if monthly_charges < 0:
    raise ValueError("Monthly charges cannot be negative.")

# Build a customer record
# Features not requested from the user use example values.
customer = pd.DataFrame([{
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": tenure,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": contract,
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": monthly_charges,
    "TotalCharges": monthly_charges * tenure
}])

# Predict churn probability
yes_index = list(model.classes_).index("Yes")
probability = model.predict_proba(customer)[0, yes_index]

# Apply our selected decision threshold
threshold = 0.35
prediction = "Yes" if probability >= threshold else "No"

# Display results
print("\n--- Customer Churn Prediction ---")
print("Predicted churn:", prediction)
print(f"Churn probability: {probability:.1%}")
print(f"Decision threshold: {threshold:.0%}")