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
print("   AGRITRUST AI - RELIABILITY MODEL TRAINING")
print("=" * 60)


# ---------------------------------------------------------
# 1. Load reliability datasets
# ---------------------------------------------------------

train_df = pd.read_csv("reliability_validation.csv")
test_df = pd.read_csv("reliability_test.csv")

print("\nTraining reliability data:", train_df.shape)
print("Testing reliability data :", test_df.shape)


# ---------------------------------------------------------
# 2. Select features and target
# ---------------------------------------------------------

features = [
    "dt_prediction",
    "dt_confidence",
    "rf_prediction",
    "rf_confidence",
    "model_agreement"
]

X_train = train_df[features]
y_train = train_df["reliable"]

X_test = test_df[features]
y_test = test_df["reliable"]

print("\nReliability features:")
print(features)


# ---------------------------------------------------------
# 3. Train Decision Tree
# ---------------------------------------------------------

print("\n" + "-" * 60)
print("Training Reliability Decision Tree...")
print("-" * 60)

dt_reliability = DecisionTreeClassifier(
    max_depth=5,
    class_weight="balanced",
    random_state=42
)

dt_reliability.fit(X_train, y_train)

dt_rel_pred = dt_reliability.predict(X_test)

dt_rel_accuracy = accuracy_score(
    y_test,
    dt_rel_pred
)

print("\nDecision Tree Reliability Accuracy:",
      round(dt_rel_accuracy * 100, 2), "%")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        dt_rel_pred,
        target_names=[
            "Potentially Unreliable",
            "Reliable"
        ]
    )
)

print("\nConfusion Matrix:")
print(confusion_matrix(
    y_test,
    dt_rel_pred
))


# ---------------------------------------------------------
# 4. Train Random Forest
# ---------------------------------------------------------

print("\n" + "-" * 60)
print("Training Reliability Random Forest...")
print("-" * 60)

rf_reliability = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

rf_reliability.fit(X_train, y_train)

rf_rel_pred = rf_reliability.predict(X_test)

rf_rel_accuracy = accuracy_score(
    y_test,
    rf_rel_pred
)

print("\nRandom Forest Reliability Accuracy:",
      round(rf_rel_accuracy * 100, 2), "%")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        rf_rel_pred,
        target_names=[
            "Potentially Unreliable",
            "Reliable"
        ]
    )
)

print("\nConfusion Matrix:")
print(confusion_matrix(
    y_test,
    rf_rel_pred
))


# ---------------------------------------------------------
# 5. Compare reliability models
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("       RELIABILITY MODEL COMPARISON")
print("=" * 60)

print(
    f"Decision Tree  : "
    f"{dt_rel_accuracy * 100:.2f}%"
)

print(
    f"Random Forest  : "
    f"{rf_rel_accuracy * 100:.2f}%"
)

if rf_rel_accuracy >= dt_rel_accuracy:
    best_reliability = rf_reliability
    best_name = "Random Forest"
else:
    best_reliability = dt_reliability
    best_name = "Decision Tree"

print(f"\nBest Reliability Model: {best_name}")


# ---------------------------------------------------------
# 6. Save reliability models
# ---------------------------------------------------------

joblib.dump(
    dt_reliability,
    "reliability_decision_tree.pkl"
)

joblib.dump(
    rf_reliability,
    "reliability_random_forest.pkl"
)

print("\nReliability models saved:")

print("  reliability_decision_tree.pkl")
print("  reliability_random_forest.pkl")


print("\n" + "=" * 60)
print("       RELIABILITY TRAINING COMPLETED")
print("=" * 60)