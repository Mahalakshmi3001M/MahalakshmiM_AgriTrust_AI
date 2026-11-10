import pandas as pd
import joblib

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


print("=" * 60)
print("   AGRITRUST AI - IMPROVED RELIABILITY TRAINING")
print("=" * 60)


# ---------------------------------------------------------
# 1. Load datasets
# ---------------------------------------------------------

train_df = pd.read_csv(
    "improved_reliability_validation.csv"
)

test_df = pd.read_csv(
    "improved_reliability_test.csv"
)

print("\nTraining data:", train_df.shape)
print("Testing data :", test_df.shape)


# ---------------------------------------------------------
# 2. Select features
# ---------------------------------------------------------

features = [
    "dt_prediction",
    "dt_confidence",
    "dt_disease_probability",
    "dt_confidence_gap",
    "dt_entropy",

    "rf_prediction",
    "rf_confidence",
    "rf_disease_probability",
    "rf_confidence_gap",
    "rf_entropy",

    "confidence_difference",
    "probability_difference",

    "model_agreement"
]


X_train = train_df[features]
y_train = train_df["reliable"]

X_test = test_df[features]
y_test = test_df["reliable"]


print("\nNumber of reliability features:",
      len(features))


# ---------------------------------------------------------
# 3. Train Decision Tree
# ---------------------------------------------------------

print("\n" + "-" * 60)
print("Training Improved Reliability Decision Tree...")
print("-" * 60)

dt_model = DecisionTreeClassifier(
    max_depth=6,
    min_samples_leaf=8,
    class_weight="balanced",
    random_state=42
)

dt_model.fit(X_train, y_train)

dt_pred = dt_model.predict(X_test)

dt_accuracy = accuracy_score(
    y_test,
    dt_pred
)

print(
    f"\nDecision Tree Accuracy: "
    f"{dt_accuracy * 100:.2f}%"
)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        dt_pred,
        target_names=[
            "Potentially Unreliable",
            "Reliable"
        ]
    )
)

print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        dt_pred
    )
)


# ---------------------------------------------------------
# 4. Train Random Forest
# ---------------------------------------------------------

print("\n" + "-" * 60)
print("Training Improved Reliability Random Forest...")
print("-" * 60)

rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    min_samples_leaf=5,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)

rf_accuracy = accuracy_score(
    y_test,
    rf_pred
)

print(
    f"\nRandom Forest Accuracy: "
    f"{rf_accuracy * 100:.2f}%"
)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        rf_pred,
        target_names=[
            "Potentially Unreliable",
            "Reliable"
        ]
    )
)

print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        rf_pred
    )
)


# ---------------------------------------------------------
# 5. Model comparison
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("       IMPROVED RELIABILITY COMPARISON")
print("=" * 60)

print(
    f"\nDecision Tree : "
    f"{dt_accuracy * 100:.2f}%"
)

print(
    f"Random Forest : "
    f"{rf_accuracy * 100:.2f}%"
)


if rf_accuracy >= dt_accuracy:

    best_model = rf_model
    best_name = "Random Forest"

else:

    best_model = dt_model
    best_name = "Decision Tree"


print(
    f"\nBest Reliability Model: "
    f"{best_name}"
)


# ---------------------------------------------------------
# 6. Save models
# ---------------------------------------------------------

joblib.dump(
    dt_model,
    "improved_reliability_decision_tree.pkl"
)

joblib.dump(
    rf_model,
    "improved_reliability_random_forest.pkl"
)


print("\nModels saved:")

print(
    "  improved_reliability_decision_tree.pkl"
)

print(
    "  improved_reliability_random_forest.pkl"
)


print("\n" + "=" * 60)
print("   IMPROVED RELIABILITY TRAINING COMPLETED")
print("=" * 60)