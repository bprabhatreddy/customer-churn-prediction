# Model Card: Customer Churn Prediction

## 1. Model Overview

- **Project:** Telco Customer Churn Prediction
- **Task:** Binary classification
- **Objective:** Estimate whether a telecommunications customer is likely to churn.
- **Model:** Scikit-learn classification pipeline
- **Application:** Streamlit web application
- **Decision threshold:** 0.35 for the predicted churn probability

## 2. Intended Use

The model is intended for educational and demonstration purposes, including exploring customer churn risk and understanding how customer characteristics relate to predictions.

It may help demonstrate how a business could prioritize customers for further investigation or retention efforts.

**The model should not be used as the sole basis for real business decisions without further validation.**

## 3. Model Evaluation

The following results are from the project's recorded test-set evaluation.

| Metric | Result |
|---|---:|
| Precision | 54.51% |
| Recall | 71.12% |
| F1-score | 61.72% |
| Decision threshold | 0.35 |
| Cross-validation F1-score | 59.75% |

### Interpretation

- **Precision:** Of the customers predicted to churn, approximately 54.51% actually churned in the evaluated test set.
- **Recall:** The model identified approximately 71.12% of the customers who actually churned.
- **F1-score:** The harmonic mean of precision and recall was approximately 61.72%.

The threshold of 0.35 prioritizes identifying more potential churners than a stricter threshold might, potentially at the cost of more false-positive predictions.

## 4. Limitations

- Predictions depend on the quality and representativeness of the training data.
- Performance may differ for customers from other populations, time periods, or telecommunications providers.
- The model may produce false positives and false negatives.
- Customer behavior can change over time, so performance should be monitored before any real-world deployment.
- Feature explanations describe the model's behavior; they do not establish that a feature causes churn.
- Test-set results alone do not establish business value or guarantee future performance.

## 5. Ethical Considerations

Predictions should support, rather than replace, human judgment. Any real-world use should assess data privacy, potential bias, and the impact of retention decisions on customers.

## 6. Validation and Testing

The project includes automated tests for:

- Loading the saved model pipeline
- Validity of predicted probabilities
- Validity of predicted classes
- Application of the configured decision threshold
- Handling of an unfamiliar categorical value

GitHub Actions runs the automated tests when changes are pushed to the `main` branch or a pull request targets `main`.

## 7. Intended Audience

Students, recruiters, and developers reviewing a machine-learning classification project.

## 8. Important Disclaimer

This is an educational machine-learning project, not a validated production decision system. Its predictions should not be treated as guarantees of customer behavior.
