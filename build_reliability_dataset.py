import numpy as np
import pandas as pd
import joblib

from sklearn.metrics import accuracy_score


print("=" * 60)
print("   AGRITRUST AI - IMPROVED RELIABILITY DATASET")
print("=" * 60)


# ---------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------

X_val = np.load("X_val.npy")
y_val = np.load("y_val.npy")

X_test = np.load("X_test.npy")
y_test = np.load("y_test.npy")

print("\nValidation data:", X_val.shape)
print("Test data      :", X_test.shape)


# ---------------------------------------------------------
# 2. Load first-stage models
# ---------------------------------------------------------

dt_model = joblib.load("decision_tree_model.pkl")
rf_model = joblib.load("random_forest_model.pkl")

print("\nFirst-stage models loaded successfully.")


# ---------------------------------------------------------
# 3. Function to calculate entropy
# ---------------------------------------------------------

def calculate_entropy(probabilities):

    probabilities = np.clip(
        probabilities,
        1e-10,
        1
    )

    return -np.sum(
        probabilities * np.log2(probabilities),
        axis=1
    )


# ---------------------------------------------------------
# 4. Function to create reliability features
# ---------------------------------------------------------

def create_reliability_features(X):

    # Predictions
    dt_prediction = dt_model.predict(X)
    rf_prediction = rf_model.predict(X)

    # Probabilities
    dt_probability = dt_model.predict_proba(X)
    rf_probability = rf_model.predict_proba(X)

    # Maximum confidence
    dt_confidence = np.max(dt_probability, axis=1)
    rf_confidence = np.max(rf_probability, axis=1)

    # Disease probability
    dt_disease_probability = dt_probability[:, 1]
    rf_disease_probability = rf_probability[:, 1]

    # Confidence margin
    dt_confidence_gap = (
        np.max(dt_probability, axis=1)
        - np.min(dt_probability, axis=1)
    )

    rf_confidence_gap = (
        np.max(rf_probability, axis=1)
        - np.min(rf_probability, axis=1)
    )

    # Difference between model confidence
    confidence_difference = np.abs(
        dt_confidence - rf_confidence
    )

    # Model agreement
    model_agreement = (
        dt_prediction == rf_prediction
    ).astype(int)

    # Probability difference
    probability_difference = np.abs(
        dt_disease_probability -
        rf_disease_probability
    )

    # Entropy
    dt_entropy = calculate_entropy(dt_probability)
    rf_entropy = calculate_entropy(rf_probability)

    # Final feature table
    data = pd.DataFrame({

        "dt_prediction":
            dt_prediction,

        "dt_confidence":
            dt_confidence,

        "dt_disease_probability":
            dt_disease_probability,

        "dt_confidence_gap":
            dt_confidence_gap,

        "dt_entropy":
            dt_entropy,

        "rf_prediction":
            rf_prediction,

        "rf_confidence":
            rf_confidence,

        "rf_disease_probability":
            rf_disease_probability,

        "rf_confidence_gap":
            rf_confidence_gap,

        "rf_entropy":
            rf_entropy,

        "confidence_difference":
            confidence_difference,

        "probability_difference":
            probability_difference,

        "model_agreement":
            model_agreement
    })

    return data


# ---------------------------------------------------------
# 5. Create validation reliability data
# ---------------------------------------------------------

print("\nCreating validation reliability features...")

validation_features = create_reliability_features(X_val)

# Target = whether Random Forest prediction is correct
validation_features["reliable"] = (
    validation_features["rf_prediction"] == y_val
).astype(int)

print("\nValidation reliability dataset:")
print(validation_features.head())

print("\nValidation label distribution:")
print(
    validation_features["reliable"]
    .value_counts()
    .rename(index={
        0: "Incorrect",
        1: "Correct"
    })
)

validation_features.to_csv(
    "improved_reliability_validation.csv",
    index=False
)

print("\nSaved:")
print("improved_reliability_validation.csv")


# ---------------------------------------------------------
# 6. Create test reliability data
# ---------------------------------------------------------

print("\nCreating test reliability features...")

test_features = create_reliability_features(X_test)

# Target
test_features["reliable"] = (
    test_features["rf_prediction"] == y_test
).astype(int)

print("\nTest reliability dataset:")
print(test_features.head())

test_features.to_csv(
    "improved_reliability_test.csv",
    index=False
)

print("\nSaved:")
print("improved_reliability_test.csv")


# ---------------------------------------------------------
# 7. First-stage validation performance
# ---------------------------------------------------------

rf_val_accuracy = accuracy_score(
    y_val,
    validation_features["rf_prediction"]
)

dt_val_accuracy = accuracy_score(
    y_val,
    validation_features["dt_prediction"]
)

print("\n" + "=" * 60)
print("       FIRST-STAGE VALIDATION RESULTS")
print("=" * 60)

print(
    f"\nDecision Tree Accuracy : "
    f"{dt_val_accuracy * 100:.2f}%"
)

print(
    f"Random Forest Accuracy : "
    f"{rf_val_accuracy * 100:.2f}%"
)

print(
    f"Model Agreement : "
    f"{validation_features['model_agreement'].mean() * 100:.2f}%"
)


print("\n" + "=" * 60)
print("   IMPROVED RELIABILITY DATASET COMPLETED")
print("=" * 60)