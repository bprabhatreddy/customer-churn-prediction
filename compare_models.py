
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

# 1. Load data
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"] = df["TotalCharges"].fillna(0)

X = df.drop(columns=["Churn", "customerID"])
y = (df["Churn"] == "Yes").astype(int)

# 2. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# 3. Preprocessing
numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
categorical_features = X.select_dtypes(exclude=["number"]).columns.tolist()

preprocessor = ColumnTransformer([
    ("numeric", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]), numeric_features),

    ("categorical", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]), categorical_features),
])

# 4. Best configuration from the previous search
model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LogisticRegression(
        C=0.1,
        max_iter=2000,
        class_weight="balanced",
    )),
])

# 5. Generate out-of-fold probabilities on training data
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("Generating training out-of-fold predictions...")

oof_probabilities = cross_val_predict(
    model,
    X_train,
    y_train,
    cv=cv,
    method="predict_proba",
    n_jobs=-1,
)[:, 1]

# Select threshold that maximizes training out-of-fold F1
thresholds = [i / 100 for i in range(10, 81, 1)]

threshold_scores = [
    f1_score(y_train, oof_probabilities >= t)
    for t in thresholds
]

best_index = threshold_scores.index(max(threshold_scores))
best_threshold = thresholds[best_index]

print(f"\nSelected threshold: {best_threshold:.2f}")
print(f"Training out-of-fold F1: {threshold_scores[best_index]:.3f}")

# 6. Fit on all training data
print("\nFitting the model on training data...")
model.fit(X_train, y_train)

test_probabilities = model.predict_proba(X_test)[:, 1]

# 7. Evaluate at both thresholds
evaluations = {
    "Original baseline (0.35)": 0.35,
    "Training-selected threshold": best_threshold,
}

results = []

for name, threshold in evaluations.items():
    predictions = (test_probabilities >= threshold).astype(int)

    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    f1 = f1_score(y_test, predictions, zero_division=0)
    auc = roc_auc_score(y_test, test_probabilities)

    results.append({
        "Evaluation": name,
        "Threshold": threshold,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "ROC-AUC": auc,
    })

    print(f"\n========== {name} ==========")
    print(f"Threshold: {threshold:.2f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1-score:  {f1:.3f}")
    print(f"ROC-AUC:   {auc:.3f}")
    print("Confusion matrix (actual rows, predicted columns):")
    print(confusion_matrix(y_test, predictions))

print("\n========== COMPARISON ==========")
print(pd.DataFrame(results).set_index("Evaluation").round(3))

print("\nNo existing model files were modified.")
