
from pathlib import Path
import json

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    make_scorer,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_predict,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ============================================================
# 1. Paths and configuration
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODEL_FILE = BASE_DIR / "churn_model.pkl"
METRICS_FILE = BASE_DIR / "model_metrics.json"

RANDOM_STATE = 42
TEST_SIZE = 0.20
THRESHOLD = 0.35


# ============================================================
# 2. Load and clean data
# ============================================================

df = pd.read_csv(DATA_FILE)

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce",
)
df["TotalCharges"] = df["TotalCharges"].fillna(0)

X = df.drop(columns=["Churn", "customerID"])
y = df["Churn"]


# ============================================================
# 3. Train-test split
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y,
)


# ============================================================
# 4. Preprocessing and model pipeline
# ============================================================

numerical_cols = X.select_dtypes(include="number").columns.tolist()
categorical_cols = X.select_dtypes(
    include=["object", "string", "category"]
).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_cols),
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_cols,
        ),
    ]
)

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000)),
    ]
)


# ============================================================
# 5. Cross-validation and hyperparameter tuning
# ============================================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE,
)

f1_churn = make_scorer(f1_score, pos_label="Yes")

grid_search = GridSearchCV(
    estimator=model,
    param_grid={"classifier__C": [0.01, 0.1, 1, 10, 100]},
    scoring=f1_churn,
    cv=cv,
    n_jobs=-1,
    refit=True,
)

print("Training and tuning Logistic Regression...")
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

print(f"Best C: {grid_search.best_params_['classifier__C']}")
print(f"Best cross-validation F1: {grid_search.best_score_:.4f}")


# ============================================================
# 6. Select threshold using training data only
# ============================================================

# Out-of-fold probabilities avoid choosing the threshold
# using predictions from models trained on those same rows.
oof_probabilities = cross_val_predict(
    best_model,
    X_train,
    y_train,
    cv=cv,
    method="predict_proba",
    n_jobs=-1,
)

yes_index = list(best_model.classes_).index("Yes")
oof_probabilities = oof_probabilities[:, yes_index]

print("\nOut-of-fold threshold comparison:")

for candidate_threshold in [0.25, 0.30, 0.35, 0.40, 0.45, 0.50]:
    oof_predictions = [
        "Yes" if probability >= candidate_threshold else "No"
        for probability in oof_probabilities
    ]

    precision = precision_score(
        y_train, oof_predictions, pos_label="Yes", zero_division=0
    )
    recall = recall_score(
        y_train, oof_predictions, pos_label="Yes", zero_division=0
    )
    f1 = f1_score(
        y_train, oof_predictions, pos_label="Yes", zero_division=0
    )

    print(
        f"Threshold: {candidate_threshold:.2f} | "
        f"Precision: {precision:.3f} | "
        f"Recall: {recall:.3f} | "
        f"F1: {f1:.3f}"
    )


# Keep the existing production threshold unchanged.
chosen_threshold = THRESHOLD


# ============================================================
# 7. Evaluate on the held-out test set
# ============================================================

# GridSearchCV has refitted best_model on the training split.
test_probabilities = best_model.predict_proba(X_test)[:, yes_index]

test_predictions = [
    "Yes" if probability >= chosen_threshold else "No"
    for probability in test_probabilities
]

test_precision = precision_score(
    y_test, test_predictions, pos_label="Yes", zero_division=0
)
test_recall = recall_score(
    y_test, test_predictions, pos_label="Yes", zero_division=0
)
test_f1 = f1_score(
    y_test, test_predictions, pos_label="Yes", zero_division=0
)

print(f"\nFinal test results at threshold {chosen_threshold:.2f}")
print(
    classification_report(
        y_test,
        test_predictions,
        labels=["No", "Yes"],
        zero_division=0,
    )
)
print("Confusion matrix (rows=true, columns=predicted; No, Yes):")
print(
    confusion_matrix(
        y_test,
        test_predictions,
        labels=["No", "Yes"],
    )
)


# ============================================================
# 8. Save model and metrics
# ============================================================

metrics = {
    "threshold": chosen_threshold,
    "best_C": float(grid_search.best_params_["classifier__C"]),
    "cv_f1": float(grid_search.best_score_),
    "test_precision": float(test_precision),
    "test_recall": float(test_recall),
    "test_f1": float(test_f1),
}

joblib.dump(best_model, MODEL_FILE)

METRICS_FILE.write_text(
    json.dumps(metrics, indent=4) + "\n",
    encoding="utf-8",
)

print(f"\nModel saved to: {MODEL_FILE}")
print(f"Metrics saved to: {METRICS_FILE}")
print("Training completed successfully.")
