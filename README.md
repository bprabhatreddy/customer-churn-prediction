# Telco Customer Churn Prediction

A machine learning project that predicts customer churn using customer account information and explains individual predictions using SHAP.

## Project Overview

Customer churn occurs when a customer stops using a company's services. This project uses a Logistic Regression model to estimate the likelihood that a telecommunications customer will leave.

The project includes an interactive Streamlit web application where users can enter customer details, receive a churn prediction, and explore the features influencing that prediction.

## Features

- Data cleaning and preprocessing
- Machine learning pipeline using Logistic Regression
- Hyperparameter tuning using GridSearchCV
- Churn probability prediction
- Adjustable classification threshold
- Individual prediction explanations using SHAP
- Interactive Streamlit interface
- Visualizations of feature contributions

## Technologies Used

- Python
- Pandas and NumPy
- Scikit-learn
- SHAP
- Streamlit
- Matplotlib
- Joblib

## Project Structure

```text
customer-churn-prediction/
├── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── customer-churn-prediction.py
├── churn_model.pkl
├── predict.py
├── app.py
├── requirements.txt
└── README.md
```

## Installation and Setup

1. Clone or download this repository.
2. Open a terminal in the project directory.
3. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Run the Streamlit application:

   ```bash
   streamlit run app.py
   ```

5. Open the local URL displayed in the terminal.

Ensure that the dataset and saved model file are available in the project directory.

## Model Information

The project uses Logistic Regression inside a Scikit-learn preprocessing pipeline. Numerical and categorical features are processed before being passed to the classifier.

The classification threshold was selected using out-of-fold predictions to balance precision and recall. The application uses a threshold of 0.35 for its churn classification.

The model's predictions are estimates and should not be treated as guarantees that a customer will leave.

## Model Explainability

SHAP is used to explain individual predictions by showing which customer features push the model's output toward or away from churn.

The application displays the strongest feature contributions in a table and a chart. These explanations describe the model's behavior; they do not establish that a feature causes customer churn.

## Limitations

- The model's performance depends on the dataset and its patterns.
- The application estimates total charges using monthly charges multiplied by tenure, so this value may differ from a customer's actual historical total.
- The model may produce false positives and false negatives.
- Predictions should support, rather than replace, business judgment.

## Future Improvements

- Compare Logistic Regression with additional classification algorithms.
- Improve the interface and add more visualizations.
- Evaluate performance on additional customer datasets.
- Deploy the application online.
