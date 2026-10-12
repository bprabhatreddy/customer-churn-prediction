import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


# =========================
# 1. Load dataset
# =========================

df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")


# =========================
# 2. Clean TotalCharges
# =========================

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

df["TotalCharges"] = df["TotalCharges"].fillna(0)


# =========================
# 3. Features and target
# =========================

X = df.drop(["Churn", "customerID"], axis=1)
y = df["Churn"]


# =========================
# 4. Train-test split
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# =========================
# 5. Identify column types
# =========================

numerical_cols = X.select_dtypes(
    include=["int64", "float64"]
).columns

categorical_cols = X.select_dtypes(
    include="string"
).columns


# =========================
# 6. Preprocessing
# =========================

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ]
)


# =========================
# 7. Model pipeline
# =========================

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000))
    ]
)


# =========================
# 8. Train model
# =========================

model.fit(X_train, y_train)


# =========================
# 9. Make predictions
# =========================

y_pred = model.predict(X_test)

# print(y_pred)

from sklearn.metrics import confusion_matrix

# print(confusion_matrix(y_test, y_pred))

from sklearn.ensemble import RandomForestClassifier


# =========================
# Random Forest model
# =========================

rf_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=200,
            random_state=42
        ))
    ]
)


# Train Random Forest
rf_model.fit(X_train, y_train)


# Predictions
rf_pred = rf_model.predict(X_test)


# # Evaluation
# print("Random Forest Results:")
# print(classification_report(y_test, rf_pred))

from sklearn.metrics import precision_score, recall_score, f1_score


# Get probability of churn (Yes)
y_prob = model.predict_proba(X_test)[:, 1]


# Try a lower threshold
threshold = 0.40

y_pred_threshold = (y_prob >= threshold).astype(int)

# Convert 0/1 back to No/Yes
y_pred_threshold = pd.Series(y_pred_threshold).map({
    0: "No",
    1: "Yes"
})


print("Threshold:", threshold)
print("Precision:", precision_score(y_test, y_pred_threshold, pos_label="Yes"))
print("Recall:", recall_score(y_test, y_pred_threshold, pos_label="Yes"))
print("F1:", f1_score(y_test, y_pred_threshold, pos_label="Yes"))

thresholds = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]

print("\nThreshold comparison:")

for threshold in thresholds:
    y_pred_threshold = (y_prob >= threshold).astype(int)

    y_pred_threshold = pd.Series(y_pred_threshold).map({
        0: "No",
        1: "Yes"
    })

    precision = precision_score(
        y_test,
        y_pred_threshold,
        pos_label="Yes"
    )

    recall = recall_score(
        y_test,
        y_pred_threshold,
        pos_label="Yes"
    )

    f1 = f1_score(
        y_test,
        y_pred_threshold,
        pos_label="Yes"
    )

    print(
        f"Threshold: {threshold:.2f} | "
        f"Precision: {precision:.3f} | "
        f"Recall: {recall:.3f} | "
        f"F1: {f1:.3f}"
    )

    from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import make_scorer, f1_score


cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

f1_churn = make_scorer(
    f1_score,
    pos_label="Yes"
)

cv_scores = cross_val_score(
    model,
    X_train,
    y_train,
    cv=cv,
    scoring=f1_churn
)

print("Cross-validation F1 scores:")
print(cv_scores)

print("Mean F1:", cv_scores.mean())
print("Standard deviation:", cv_scores.std())

rf_cv_scores = cross_val_score(
    rf_model,
    X_train,
    y_train,
    cv=cv,
    scoring=f1_churn
)

print("\nRandom Forest CV F1 scores:")
print(rf_cv_scores)

print("Random Forest Mean F1:", rf_cv_scores.mean())
print("Random Forest Standard deviation:", rf_cv_scores.std())

from sklearn.model_selection import GridSearchCV


param_grid = {
    "classifier__C": [0.01, 0.1, 1, 10, 100]
}


grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=cv,
    scoring=f1_churn,
    n_jobs=-1
)


grid_search.fit(X_train, y_train)


print("\nBest C:", grid_search.best_params_["classifier__C"])
print("Best CV F1:", grid_search.best_score_)

results = pd.DataFrame(grid_search.cv_results_)

print(
    results[[
        "param_classifier__C",
        "mean_test_score",
        "std_test_score",
        "rank_test_score"
    ]].sort_values("rank_test_score")
)

from sklearn.model_selection import cross_val_predict
from sklearn.base import clone
import numpy as np

best_model = grid_search.best_estimator_

oof_probabilities = cross_val_predict(
    best_model,
    X_train,
    y_train,
    cv=cv,
    method="predict_proba",
    n_jobs=-1
)

yes_index = list(best_model.classes_).index("Yes")
oof_probabilities = oof_probabilities[:, yes_index]

print("Number of out-of-fold probabilities:", len(oof_probabilities))
print("First 10 probabilities:", oof_probabilities[:10])

from sklearn.metrics import precision_score, recall_score, f1_score

print("\nOut-of-fold threshold comparison:")

for threshold in [0.25, 0.30, 0.35, 0.40, 0.45, 0.50]:
    predictions = (oof_probabilities >= threshold).astype(int)

    predictions = pd.Series(predictions).map({
        0: "No",
        1: "Yes"
    })

    precision = precision_score(y_train, predictions, pos_label="Yes")
    recall = recall_score(y_train, predictions, pos_label="Yes")
    f1 = f1_score(y_train, predictions, pos_label="Yes")

    print(
        f"Threshold: {threshold:.2f} | "
        f"Precision: {precision:.3f} | "
        f"Recall: {recall:.3f} | "
        f"F1: {f1:.3f}"
    )

    from sklearn.metrics import classification_report, confusion_matrix

final_model = grid_search.best_estimator_
final_model.fit(X_train, y_train)

yes_index = list(final_model.classes_).index("Yes")
test_probabilities = final_model.predict_proba(X_test)[:, yes_index]

chosen_threshold = 0.35
y_test_pred = pd.Series(
    (test_probabilities >= chosen_threshold).astype(int)
).map({0: "No", 1: "Yes"})

print("\nFinal Model — Threshold:", chosen_threshold)
print(classification_report(y_test, y_test_pred))
print("Confusion matrix:")
print(confusion_matrix(y_test, y_test_pred, labels=["No", "Yes"]))

import joblib

joblib.dump(final_model, "churn_model.pkl")

print("Model saved successfully!")

import joblib
import pandas as pd

# Load the saved model
loaded_model = joblib.load("churn_model.pkl")

# Example customer
new_customer = pd.DataFrame([{
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 2,
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
    "MonthlyCharges": 85.5,
    "TotalCharges": 171.0
}])

# Predict churn
prediction = loaded_model.predict(new_customer)[0]

# Get probability of churn
yes_index = list(loaded_model.classes_).index("Yes")
churn_probability = loaded_model.predict_proba(new_customer)[0, yes_index]

print("Predicted churn:", prediction)
print(f"Churn probability: {churn_probability:.1%}")