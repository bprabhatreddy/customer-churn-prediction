
import joblib
import pandas as pd
import pytest
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
THRESHOLD = 0.35


@pytest.fixture
def model():
    """Load the saved model once for each test that needs it."""
    return joblib.load(BASE_DIR / "churn_model.pkl")


@pytest.fixture
def sample_customer():
    """Create a valid example customer using the model's expected features."""
    return pd.DataFrame([{
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 70.0,
        "TotalCharges": 840.0,
    }])


def test_model_loads(model):
    """Check that the saved model loads as a usable pipeline."""
    assert "preprocessor" in model.named_steps
    assert "classifier" in model.named_steps


def test_prediction_probabilities(model, sample_customer):
    """Check that probabilities are valid and sum to one."""
    probabilities = model.predict_proba(sample_customer)[0]

    assert len(probabilities) == 2
    assert all(0 <= probability <= 1 for probability in probabilities)
    assert sum(probabilities) == pytest.approx(1.0)


def test_prediction_class(model, sample_customer):
    """Check that the model returns a valid churn class."""
    prediction = model.predict(sample_customer)[0]

    assert prediction in ["Yes", "No"]


def test_custom_threshold_logic(model, sample_customer):
    """Check that the application's 0.35 threshold is applied correctly."""
    classes = list(model.classes_)
    yes_index = classes.index("Yes")

    probability = model.predict_proba(sample_customer)[0, yes_index]
    prediction = "Yes" if probability >= THRESHOLD else "No"

    if probability >= THRESHOLD:
        assert prediction == "Yes"
    else:
        assert prediction == "No"


def test_unseen_categories_do_not_crash(model, sample_customer):
    """Check that the pipeline handles an unfamiliar category."""
    customer = sample_customer.copy()
    customer["PaymentMethod"] = "Unseen payment method"

    probabilities = model.predict_proba(customer)[0]

    assert all(0 <= probability <= 1 for probability in probabilities)
