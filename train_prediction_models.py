import numpy as np
import joblib

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


print("=" * 60)
print("       AGRITRUST AI - MODEL TRAINING")
print("=" * 60)

# ---------------------------------------------------------
# 1. Load extracted features
# ---------------------------------------------------------

X_train = np.load("X_train.npy")
y_train = np.load("y_train.npy")

X_val = np.load("X_val.npy")
y_val = np.load("y_val.npy")

X_test = np.load("X_test.npy")
y_test = np.load("y_test.npy")

print("\nFeature data loaded successfully.")

print("Training shape   :", X_train.shape)
print("Validation shape :", X_val.shape)
print("Testing shape    :", X_test.shape)


# ---------------------------------------------------------
# 2. Decision Tree
# ---------------------------------------------------------

print("\n" + "-" * 60)
print("Training Decision Tree...")
print("-" * 60)

dt_model = DecisionTreeClassifier(
    max_depth=20,
    class_weight="balanced",
    random_state=42
)

dt_model.fit(X_train, y_train)

dt_pred = dt_model.predict(X_test)

dt_accuracy = accuracy_score(y_test, dt_pred)

print("\nDecision Tree Accuracy:",
      round(dt_accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(
    y_test,
    dt_pred,
    target_names=["Healthy", "Disease"]
))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, dt_pred))


# ---------------------------------------------------------
# 3. Random Forest
# ---------------------------------------------------------

print("\n" + "-" * 60)
print("Training Random Forest...")
print("-" * 60)

rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)

rf_accuracy = accuracy_score(y_test, rf_pred)

print("\nRandom Forest Accuracy:",
      round(rf_accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(
    y_test,
    rf_pred,
    target_names=["Healthy", "Disease"]
))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, rf_pred))


# ---------------------------------------------------------
# 4. Compare Models
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("             MODEL COMPARISON")
print("=" * 60)

print(f"Decision Tree  : {dt_accuracy * 100:.2f}%")
print(f"Random Forest  : {rf_accuracy * 100:.2f}%")

if rf_accuracy > dt_accuracy:
    print("\nBest Model: Random Forest")
else:
    print("\nBest Model: Decision Tree")


# ---------------------------------------------------------
# 5. Save models
# ---------------------------------------------------------

joblib.dump(dt_model, "decision_tree_model.pkl")
joblib.dump(rf_model, "random_forest_model.pkl")

print("\nModels saved successfully:")
print("  decision_tree_model.pkl")
print("  random_forest_model.pkl")

print("\n" + "=" * 60)
print("           TRAINING COMPLETED")
print("=" * 60)