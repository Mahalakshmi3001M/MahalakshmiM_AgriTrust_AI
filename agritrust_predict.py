import numpy as np
import joblib

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image


# ============================================================
#                    AGRITRUST AI
#       CROP + CONDITION + RELIABILITY PREDICTION
# ============================================================

print("=" * 60)
print("                 AGRITRUST AI")
print("      CROP + CONDITION + RELIABILITY PREDICTION")
print("=" * 60)


# ============================================================
# 1. LOAD MOBILE NET V2
# ============================================================

print("\nLoading MobileNetV2...")

cnn = MobileNetV2(
    weights="imagenet",
    include_top=False,
    pooling="avg",
    input_shape=(224, 224, 3)
)

print("MobileNetV2 loaded successfully.")


# ============================================================
# 2. LOAD TRAINED MODELS
# ============================================================

print("\nLoading trained models...")

crop_dt_model = joblib.load(
    "crop_decision_tree_model.pkl"
)

crop_rf_model = joblib.load(
    "crop_random_forest_model.pkl"
)

crop_labels = joblib.load(
    "crop_labels.pkl"
)

disease_dt_model = joblib.load(
    "decision_tree_model.pkl"
)

disease_rf_model = joblib.load(
    "random_forest_model.pkl"
)

reliability_model = joblib.load(
    "improved_reliability_random_forest.pkl"
)

print("All models loaded successfully.")


# ============================================================
# 3. GET IMAGE PATH
# ============================================================

image_path = input(
    "\nEnter leaf image path: "
).strip()


# ============================================================
# 4. LOAD IMAGE
# ============================================================

print("\nLoading image...")

img = image.load_img(
    image_path,
    target_size=(224, 224)
)

img_array = image.img_to_array(img)

img_array = np.expand_dims(
    img_array,
    axis=0
)

img_array = preprocess_input(
    img_array
)

print("Image loaded successfully.")


# ============================================================
# 5. EXTRACT FEATURES
# ============================================================

print("\nExtracting image features...")

features = cnn.predict(
    img_array,
    verbose=0
)

print("Feature extraction completed.")


# ============================================================
# 6. CROP PREDICTION
# ============================================================

print("\n" + "-" * 60)
print("CROP PREDICTION")
print("-" * 60)


crop_dt_prediction = crop_dt_model.predict(
    features
)[0]

crop_rf_prediction = crop_rf_model.predict(
    features
)[0]


crop_dt_probabilities = (
    crop_dt_model.predict_proba(features)[0]
)

crop_rf_probabilities = (
    crop_rf_model.predict_proba(features)[0]
)


crop_dt_confidence = (
    np.max(crop_dt_probabilities) * 100
)

crop_rf_confidence = (
    np.max(crop_rf_probabilities) * 100
)


# Convert numeric label to crop name

crop_dt_name = [
    name
    for name, value in crop_labels.items()
    if value == crop_dt_prediction
][0]

crop_rf_name = [
    name
    for name, value in crop_labels.items()
    if value == crop_rf_prediction
][0]


crop_agreement = (
    crop_dt_prediction ==
    crop_rf_prediction
)


print(
    f"\nDecision Tree Crop       : "
    f"{crop_dt_name.title()}"
)

print(
    f"Decision Tree Confidence : "
    f"{crop_dt_confidence:.2f}%"
)

print(
    f"\nRandom Forest Crop       : "
    f"{crop_rf_name.title()}"
)

print(
    f"Random Forest Confidence : "
    f"{crop_rf_confidence:.2f}%"
)

print(
    "\nCrop Model Agreement     : "
    + ("YES" if crop_agreement else "NO")
)


# Random Forest is the final crop model

final_crop = crop_rf_name
final_crop_confidence = crop_rf_confidence


# ============================================================
# 7. CONDITION PREDICTION
# ============================================================

print("\n" + "-" * 60)
print("HEALTHY / DISEASE PREDICTION")
print("-" * 60)


dt_prediction = disease_dt_model.predict(
    features
)[0]

rf_prediction = disease_rf_model.predict(
    features
)[0]


dt_probabilities = (
    disease_dt_model.predict_proba(features)[0]
)

rf_probabilities = (
    disease_rf_model.predict_proba(features)[0]
)


dt_confidence = (
    np.max(dt_probabilities) * 100
)

rf_confidence = (
    np.max(rf_probabilities) * 100
)


dt_disease_probability = (
    dt_probabilities[1]
)

rf_disease_probability = (
    rf_probabilities[1]
)


dt_condition = (
    "Disease"
    if dt_prediction == 1
    else "Healthy"
)

rf_condition = (
    "Disease"
    if rf_prediction == 1
    else "Healthy"
)


model_agreement = (
    dt_prediction == rf_prediction
)


print(
    f"\nDecision Tree Condition  : "
    f"{dt_condition}"
)

print(
    f"Decision Tree Confidence : "
    f"{dt_confidence:.2f}%"
)

print(
    f"\nRandom Forest Condition  : "
    f"{rf_condition}"
)

print(
    f"Random Forest Confidence : "
    f"{rf_confidence:.2f}%"
)

print(
    "\nModel Agreement          : "
    + ("YES" if model_agreement else "NO")
)


# ============================================================
# 8. RELIABILITY FEATURES
# ============================================================

def calculate_entropy(probabilities):

    probabilities = np.clip(
        probabilities,
        1e-10,
        1
    )

    return -np.sum(
        probabilities *
        np.log2(probabilities)
    )


dt_confidence_gap = abs(
    2 * dt_disease_probability - 1
)

rf_confidence_gap = abs(
    2 * rf_disease_probability - 1
)

dt_entropy = calculate_entropy(
    dt_probabilities
)

rf_entropy = calculate_entropy(
    rf_probabilities
)

confidence_difference = abs(
    dt_confidence -
    rf_confidence
) / 100

probability_difference = abs(
    dt_disease_probability -
    rf_disease_probability
)

agreement_value = (
    1 if model_agreement else 0
)


# ============================================================
# 9. AI RELIABILITY PREDICTION
# ============================================================

print("\n" + "-" * 60)
print("AI RELIABILITY PREDICTION")
print("-" * 60)


reliability_features = np.array([[
    dt_prediction,
    dt_confidence / 100,
    dt_disease_probability,
    dt_confidence_gap,
    dt_entropy,

    rf_prediction,
    rf_confidence / 100,
    rf_disease_probability,
    rf_confidence_gap,
    rf_entropy,

    confidence_difference,
    probability_difference,
    agreement_value
]])


reliability_prediction = (
    reliability_model.predict(
        reliability_features
    )[0]
)

reliability_probabilities = (
    reliability_model.predict_proba(
        reliability_features
    )[0]
)


reliability_confidence = (
    np.max(reliability_probabilities)
    * 100
)


if reliability_prediction == 1:

    reliability_status = (
        "RELIABLE"
    )

else:

    reliability_status = (
        "POTENTIALLY UNRELIABLE"
    )


print(
    f"\nReliability Status       : "
    f"{reliability_status}"
)

print(
    f"Reliability Confidence   : "
    f"{reliability_confidence:.2f}%"
)


# ============================================================
# 10. FINAL RESULT
# ============================================================

print("\n" + "=" * 60)
print("                    FINAL RESULT")
print("=" * 60)

print(
    f"\nCrop                      : "
    f"{final_crop.title()}"
)

print(
    f"Crop Confidence           : "
    f"{final_crop_confidence:.2f}%"
)

print(
    f"\nCondition                 : "
    f"{rf_condition}"
)

print(
    f"Condition Confidence      : "
    f"{rf_confidence:.2f}%"
)

print(
    f"\nDecision Tree Prediction  : "
    f"{dt_condition}"
)

print(
    f"Decision Tree Confidence  : "
    f"{dt_confidence:.2f}%"
)

print(
    f"\nRandom Forest Prediction  : "
    f"{rf_condition}"
)

print(
    f"Random Forest Confidence  : "
    f"{rf_confidence:.2f}%"
)

print(
    f"\nModel Agreement           : "
    + ("YES" if model_agreement else "NO")
)

print(
    f"\nAI Reliability            : "
    f"{reliability_status}"
)

print(
    f"Reliability Confidence    : "
    f"{reliability_confidence:.2f}%"
)

print("\n" + "=" * 60)
print("              PREDICTION COMPLETED")
print("=" * 60)