import os
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image


# ============================================================
# AGRITRUST AI - CROP CLASSIFICATION
# ============================================================

print("=" * 60)
print("        AGRITRUST AI - CROP CLASSIFICATION")
print("=" * 60)


# ------------------------------------------------------------
# 1. DATASET PATH
# ------------------------------------------------------------

DATASET_PATH = os.path.join(
    "Indian_Crop_Disease_Dataset",
    "Crop Dataset"
)


# ------------------------------------------------------------
# 2. CROP LABELS
# ------------------------------------------------------------

crop_labels = {
    "coffee": 0,
    "cotton": 1,
    "jute": 2,
    "rice": 3,
    "sugarcane": 4,
    "wheat": 5
}

label_to_crop = {
    value: key
    for key, value in crop_labels.items()
}


# ------------------------------------------------------------
# 3. COLLECT ALL IMAGES RECURSIVELY
# ------------------------------------------------------------

image_paths = []
labels = []

for crop, label in crop_labels.items():

    crop_path = os.path.join(
        DATASET_PATH,
        crop
    )

    for root, dirs, files in os.walk(crop_path):

        for filename in files:

            if filename.lower().endswith(
                (".jpg", ".jpeg", ".png", ".bmp", ".webp")
            ):

                file_path = os.path.join(
                    root,
                    filename
                )

                image_paths.append(file_path)
                labels.append(label)


print("\nTotal images found:", len(image_paths))


# ------------------------------------------------------------
# 4. CREATE DATAFRAME
# ------------------------------------------------------------

df = pd.DataFrame({
    "image_path": image_paths,
    "crop": labels
})

df["crop_name"] = df["crop"].map(label_to_crop)


print("\nCrop distribution:")
print(df["crop_name"].value_counts())


# ------------------------------------------------------------
# 5. CHECK DATASET SIZE
# ------------------------------------------------------------

if len(df) != 6048:

    print(
        f"\nWARNING: Expected 6048 images, "
        f"but found {len(df)} images."
    )

else:

    print("\nDataset check: 6048 images found successfully.")


# ------------------------------------------------------------
# 6. TRAIN / VALIDATION / TEST SPLIT
# ------------------------------------------------------------

X_train, X_temp, y_train, y_temp = train_test_split(
    df["image_path"],
    df["crop"],
    test_size=0.30,
    random_state=42,
    stratify=df["crop"]
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)


print("\nDataset split:")
print("Training   :", len(X_train))
print("Validation :", len(X_val))
print("Testing    :", len(X_test))
print("Total      :", len(X_train) + len(X_val) + len(X_test))


# ------------------------------------------------------------
# 7. LOAD MOBILENETV2
# ------------------------------------------------------------

print("\nLoading MobileNetV2...")

cnn = MobileNetV2(
    weights="imagenet",
    include_top=False,
    pooling="avg",
    input_shape=(224, 224, 3)
)

print("MobileNetV2 loaded successfully.")


# ------------------------------------------------------------
# 8. FEATURE EXTRACTION
# ------------------------------------------------------------

def extract_features(image_paths):

    features = []

    for i, image_path in enumerate(image_paths):

        img = image.load_img(
            image_path,
            target_size=(224, 224)
        )

        img_array = image.img_to_array(img)

        img_array = np.expand_dims(
            img_array,
            axis=0
        )

        img_array = preprocess_input(img_array)

        feature = cnn.predict(
            img_array,
            verbose=0
        )

        features.append(feature[0])

        if (i + 1) % 500 == 0:
            print(
                f"Processed {i + 1}/{len(image_paths)} images"
            )

    return np.array(features)


# ------------------------------------------------------------
# 9. EXTRACT FEATURES
# ------------------------------------------------------------

print("\nExtracting training features...")
X_train_features = extract_features(
    X_train.tolist()
)

print("\nExtracting validation features...")
X_val_features = extract_features(
    X_val.tolist()
)

print("\nExtracting testing features...")
X_test_features = extract_features(
    X_test.tolist()
)


print("\nFeature shapes:")
print("Training   :", X_train_features.shape)
print("Validation :", X_val_features.shape)
print("Testing    :", X_test_features.shape)


# ------------------------------------------------------------
# 10. SAVE FEATURES
# ------------------------------------------------------------

np.save(
    "crop_X_train.npy",
    X_train_features
)

np.save(
    "crop_X_val.npy",
    X_val_features
)

np.save(
    "crop_X_test.npy",
    X_test_features
)

np.save(
    "crop_y_train.npy",
    y_train
)

np.save(
    "crop_y_val.npy",
    y_val
)

np.save(
    "crop_y_test.npy",
    y_test
)


# ------------------------------------------------------------
# 11. DECISION TREE
# ------------------------------------------------------------

print("\n" + "-" * 60)
print("Training Crop Decision Tree...")
print("-" * 60)

dt_model = DecisionTreeClassifier(
    max_depth=20,
    class_weight="balanced",
    random_state=42
)

dt_model.fit(
    X_train_features,
    y_train
)

dt_pred = dt_model.predict(
    X_test_features
)

dt_accuracy = accuracy_score(
    y_test,
    dt_pred
)

print(
    f"\nDecision Tree Accuracy: "
    f"{dt_accuracy * 100:.2f}%"
)

print("\nDecision Tree Classification Report:")

print(
    classification_report(
        y_test,
        dt_pred,
        target_names=list(crop_labels.keys())
    )
)


# ------------------------------------------------------------
# 12. RANDOM FOREST
# ------------------------------------------------------------

print("\n" + "-" * 60)
print("Training Crop Random Forest...")
print("-" * 60)

rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

rf_model.fit(
    X_train_features,
    y_train
)

rf_pred = rf_model.predict(
    X_test_features
)

rf_accuracy = accuracy_score(
    y_test,
    rf_pred
)

print(
    f"\nRandom Forest Accuracy: "
    f"{rf_accuracy * 100:.2f}%"
)

print("\nRandom Forest Classification Report:")

print(
    classification_report(
        y_test,
        rf_pred,
        target_names=list(crop_labels.keys())
    )
)


# ------------------------------------------------------------
# 13. CONFUSION MATRICES
# ------------------------------------------------------------

print("\nDecision Tree Confusion Matrix:")

print(
    confusion_matrix(
        y_test,
        dt_pred
    )
)

print("\nRandom Forest Confusion Matrix:")

print(
    confusion_matrix(
        y_test,
        rf_pred
    )
)


# ------------------------------------------------------------
# 14. SAVE MODELS
# ------------------------------------------------------------

joblib.dump(
    dt_model,
    "crop_decision_tree_model.pkl"
)

joblib.dump(
    rf_model,
    "crop_random_forest_model.pkl"
)

joblib.dump(
    crop_labels,
    "crop_labels.pkl"
)


# ------------------------------------------------------------
# 15. FINAL RESULT
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("          CROP CLASSIFICATION COMPLETE")
print("=" * 60)

print(
    f"\nDecision Tree Accuracy : "
    f"{dt_accuracy * 100:.2f}%"
)

print(
    f"Random Forest Accuracy : "
    f"{rf_accuracy * 100:.2f}%"
)

if rf_accuracy >= dt_accuracy:

    print("\nBest Crop Model: Random Forest")

else:

    print("\nBest Crop Model: Decision Tree")

print("\nSaved files:")
print("crop_decision_tree_model.pkl")
print("crop_random_forest_model.pkl")
print("crop_labels.pkl")

print("\n" + "=" * 60)