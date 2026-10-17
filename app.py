
import json
from pathlib import Path

import streamlit as st
import joblib
import pandas as pd
import shap


# --------------------------------------------------
# Load model and saved settings
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

MODEL_FILE = BASE_DIR / "churn_model.pkl"
METRICS_FILE = BASE_DIR / "model_metrics.json"
DATA_FILE = BASE_DIR / "WA_Fn-UseC_-Telco-Customer-Churn.csv"

if not MODEL_FILE.exists():
    st.error("Model file 'churn_model.pkl' was not found.")
    st.stop()

model = joblib.load(MODEL_FILE)

# Read the threshold saved by the training script.
# Fall back to 0.35 if the metrics file is unavailable.
threshold = 0.35

if METRICS_FILE.exists():
    try:
        with METRICS_FILE.open("r", encoding="utf-8") as file:
            metrics = json.load(file)

        threshold = float(metrics.get("threshold", 0.35))
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        st.warning(
            "Could not read model_metrics.json. "
            "Using the default threshold of 0.35."
        )

if not 0 < threshold < 1:
    st.error("The saved prediction threshold must be between 0 and 1.")
    st.stop()


# --------------------------------------------------
# Create and cache the SHAP explainer
# --------------------------------------------------

@st.cache_resource
def load_shap_explainer():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {DATA_FILE.name}"
        )

    data = pd.read_csv(DATA_FILE)

    data["TotalCharges"] = pd.to_numeric(
        data["TotalCharges"], errors="coerce"
    ).fillna(0)

    background = data.drop(
        columns=["customerID", "Churn"]
    ).sample(n=100, random_state=42)

    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]

    background_processed = preprocessor.transform(background)

    explainer = shap.LinearExplainer(
        classifier,
        background_processed
    )

    return explainer, preprocessor


try:
    explainer, preprocessor = load_shap_explainer()
except Exception as error:
    st.error(f"Could not initialize the SHAP explainer: {error}")
    st.stop()


# --------------------------------------------------
# Page
# --------------------------------------------------

st.title("Customer Churn Predictor")
st.write("Predict whether a customer is likely to leave.")

st.caption(
    f"Prediction threshold: {threshold:.0%}. "
    "Results are estimates, not guarantees."
)


# --------------------------------------------------
# Customer input form
# --------------------------------------------------

with st.form("customer_form"):

    st.subheader("Customer Details")

    col1, col2 = st.columns(2)

    with col1:
        tenure = st.number_input(
            "Customer tenure (months)",
            min_value=0,
            max_value=72,
            value=12
        )

    with col2:
        monthly_charges = st.number_input(
            "Monthly charges",
            min_value=0.0,
            max_value=200.0,
            value=70.0,
            step=5.0
        )

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    senior_citizen = st.selectbox(
        "Is the customer a senior citizen?",
        ["No", "Yes"]
    )

    partner_checked = st.checkbox(
        "Does the customer have a partner?",
        value=True
    )
    partner = "Yes" if partner_checked else "No"

    dependents = st.selectbox(
        "Does the customer have dependents?",
        ["No", "Yes"]
    )

    st.divider()
    st.subheader("Services")

    internet_service = st.selectbox(
        "Internet service type",
        ["DSL", "Fiber optic", "No"]
    )

    phone_service = st.selectbox(
        "Does the customer have phone service?",
        ["Yes", "No"]
    )

    multiple_lines = st.selectbox(
        "Multiple phone lines?",
        ["No", "Yes", "No phone service"]
    )

    online_security = st.selectbox(
        "Online security?",
        ["No", "Yes", "No internet service"]
    )

    online_backup = st.selectbox(
        "Online backup?",
        ["No", "Yes", "No internet service"]
    )

    device_protection = st.selectbox(
        "Device protection?",
        ["No", "Yes", "No internet service"]
    )

    tech_support = st.selectbox(
        "Tech support?",
        ["No", "Yes", "No internet service"]
    )

    streaming_tv = st.selectbox(
        "Streaming TV?",
        ["No", "Yes", "No internet service"]
    )

    streaming_movies = st.selectbox(
        "Streaming movies?",
        ["No", "Yes", "No internet service"]
    )

    st.divider()
    st.subheader("Billing")

    contract = st.selectbox(
        "Contract type",
        ["Month-to-month", "One year", "Two year"]
    )

    paperless_billing = st.selectbox(
        "Paperless billing?",
        ["Yes", "No"]
    )

    payment_method = st.selectbox(
        "Payment method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )

    submitted = st.form_submit_button("Predict Churn")


# --------------------------------------------------
# Prediction and explanation
# --------------------------------------------------

if submitted:

    customer = pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": 1 if senior_citizen == "Yes" else 0,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        # Approximation: actual total charges may differ.
        "TotalCharges": monthly_charges * tenure
    }])

    # Calculate churn probability for the "Yes" class.
    yes_index = list(model.classes_).index("Yes")
    probability = float(
        model.predict_proba(customer)[0, yes_index]
    )

    # Use the threshold loaded from model_metrics.json.
    prediction = "Yes" if probability >= threshold else "No"

    st.divider()
    st.subheader("Prediction Result")

    st.metric(
        "Estimated churn probability",
        f"{probability:.1%}"
    )

    st.progress(probability)

    # Risk bands are for display; the prediction threshold
    # above determines the Yes/No classification.
    if probability >= 0.60:
        st.error("Churn likelihood: High")
    elif probability >= 0.35:
        st.warning("Churn likelihood: Medium")
    else:
        st.success("Churn likelihood: Low")

    st.caption(
        "This is a model estimate, not a guarantee of what "
        "the customer will do."
    )

    if prediction == "Yes":
        st.warning("This customer is predicted to churn.")
    else:
        st.success("This customer is predicted not to churn.")

    # --------------------------------------------------
    # SHAP feature contributions
    # --------------------------------------------------

    customer_processed = preprocessor.transform(customer)
    shap_values = explainer(customer_processed)

    feature_names = preprocessor.get_feature_names_out()

    contributions = pd.DataFrame({
        "Feature": feature_names,
        "SHAP value": shap_values.values[0]
    })

    contributions["Absolute contribution"] = (
        contributions["SHAP value"].abs()
    )

    contributions = contributions.sort_values(
        "Absolute contribution",
        ascending=False
    )

    feature_labels = {
        "num__tenure": "Customer Tenure",
        "num__TotalCharges": "Total Charges",
        "num__MonthlyCharges": "Monthly Charges",
        "num__SeniorCitizen": "Senior Citizen",
        "cat__gender_Female": "Gender: Female",
        "cat__gender_Male": "Gender: Male",
        "cat__Partner_No": "No Partner",
        "cat__Partner_Yes": "Has Partner",
        "cat__Dependents_No": "No Dependents",
        "cat__Dependents_Yes": "Has Dependents",
        "cat__InternetService_Fiber optic": "Fiber Optic Internet",
        "cat__InternetService_DSL": "DSL Internet",
        "cat__InternetService_No": "No Internet Service",
        "cat__Contract_Month-to-month": "Month-to-Month Contract",
        "cat__Contract_One year": "One-Year Contract",
        "cat__Contract_Two year": "Two-Year Contract",
        "cat__StreamingTV_Yes": "Streaming TV",
        "cat__StreamingMovies_Yes": "Streaming Movies",
        "cat__PaperlessBilling_Yes": "Paperless Billing: Yes",
        "cat__PaperlessBilling_No": "Paperless Billing: No",
        "cat__PaymentMethod_Electronic check": "Electronic Check",
        "cat__OnlineSecurity_No": "No Online Security",
        "cat__TechSupport_No": "No Tech Support"
    }

    def readable_feature(name):
        return feature_labels.get(
            name,
            name.replace("cat__", "")
                .replace("num__", "")
                .replace("_", " ")
        )

    st.divider()
    st.subheader("What Influenced This Prediction?")

    top_features = contributions.head(5).copy()

    top_features["Feature"] = top_features["Feature"].map(
        readable_feature
    )

    top_features["Direction"] = top_features["SHAP value"].apply(
        lambda value: "Pushes toward churn"
        if value > 0 else
        "Pushes away from churn"
        if value < 0 else
        "No effect"
    )

    top_features = top_features.rename(columns={
        "Feature": "Customer Feature",
        "SHAP value": "Contribution"
    })

    st.dataframe(
        top_features[
            ["Customer Feature", "Contribution", "Direction"]
        ],
        hide_index=True,
        use_container_width=True
    )

    st.caption(
        "Positive contributions push the model's raw score toward "
        "churn; negative contributions push it away. SHAP values "
        "are not direct percentage-point changes in probability."
    )

    st.write("### Contribution Strength")

    chart_features = contributions.head(10).copy()
    chart_features["Feature"] = chart_features["Feature"].map(
        readable_feature
    )

    chart_data = chart_features.set_index("Feature")[["SHAP value"]]
    st.bar_chart(chart_data)

    # --------------------------------------------------
    # Dataset-based insights
    # --------------------------------------------------

    st.divider()
    st.subheader("Customer Insights")

    if contract == "Month-to-month":
        st.write(
            "This customer has a month-to-month contract. "
            "Customers with this contract type had higher churn "
            "rates in our dataset."
        )
    elif contract == "One year":
        st.write(
            "This customer has a one-year contract. Customers "
            "with one-year contracts had lower churn rates than "
            "month-to-month customers in our dataset."
        )
    else:
        st.write(
            "This customer has a two-year contract. Customers "
            "with two-year contracts had the lowest churn rate "
            "among the contract types in our dataset."
        )

    if tenure <= 12:
        st.write(
            "The customer is relatively new. Customers with "
            "12 months or less of tenure had higher churn rates "
            "in our dataset."
        )
    else:
        st.write(
            "The customer has been with the company for over "
            "12 months. Longer tenure was associated with lower "
            "churn in our dataset."
        )

    # --------------------------------------------------
    # Customer summary
    # --------------------------------------------------

    st.divider()
    st.subheader("Customer Summary")

    st.write(f"**Contract type:** {contract}")
    st.write(f"**Customer tenure:** {tenure} months")
    st.write(f"**Monthly charges:** ₹{monthly_charges:.2f}")
    st.write(
        f"**Estimated total charges:** "
        f"₹{monthly_charges * tenure:.2f}"
    )
