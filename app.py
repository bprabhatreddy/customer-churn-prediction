
import streamlit as st
import joblib
import pandas as pd
import shap

# Load the trained model
model = joblib.load("churn_model.pkl")


# Cache the SHAP explainer so it is not recreated for every prediction
@st.cache_resource
def load_shap_explainer():
    data = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

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


explainer, preprocessor = load_shap_explainer()


# Page title
st.title("Customer Churn Predictor")
st.write("Predict whether a customer is likely to leave.")


# Collect customer information in a form
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


# Predict only after the form is submitted
if submitted:

    # Build the customer input
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
        "TotalCharges": monthly_charges * tenure
    }])

    # Predict churn probability
    yes_index = list(model.classes_).index("Yes")
    probability = model.predict_proba(customer)[0, yes_index]

    # Apply the selected prediction threshold
    threshold = 0.35
    prediction = "Yes" if probability >= threshold else "No"

    # Display prediction results
    st.divider()
    st.subheader("Prediction Result")

    st.metric(
        "Estimated churn probability",
        f"{probability:.1%}"
    )
    st.progress(float(probability))

    if probability >= 0.60:
        st.error("Churn likelihood: High")
    elif probability >= 0.35:
        st.warning("Churn likelihood: Medium")
    else:
        st.success("Churn likelihood: Low")

    st.caption(
        "This is a model estimate, not a guarantee of what the customer will do."
    )

    if prediction == "Yes":
        st.warning("This customer is predicted to churn.")
    else:
        st.success("This customer is predicted not to churn.")

    # Explain this customer's prediction using the cached SHAP explainer
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

    # Translate encoded feature names into readable labels
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
            name.replace("cat__", "").replace("num__", "").replace("_", " ")
        )

    # Display the top five feature contributions
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
        width="stretch"
    )

    st.caption(
        "Positive contributions push the model's raw score toward churn; "
        "negative contributions push it away. These are not direct "
        "percentage-point changes in churn probability."
    )

    # Display a bar chart of the ten strongest contributions
    st.write("### Contribution Strength")

    chart_features = contributions.head(10).copy()
    chart_features["Feature"] = chart_features["Feature"].map(
        readable_feature
    )

    chart_data = chart_features.set_index("Feature")[["SHAP value"]]
    st.bar_chart(chart_data)

    # General insights based on patterns in the dataset
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
            "This customer has a one-year contract. "
            "Customers with one-year contracts had lower churn "
            "rates than month-to-month customers in our dataset."
        )
    else:
        st.write(
            "This customer has a two-year contract. "
            "Customers with two-year contracts had the lowest "
            "churn rate among the contract types in our dataset."
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

    # Customer summary
    st.divider()
    st.subheader("Customer Summary")

    st.write(f"**Contract type:** {contract}")
    st.write(f"**Customer tenure:** {tenure} months")
    st.write(f"**Monthly charges:** ₹{monthly_charges:.2f}")
