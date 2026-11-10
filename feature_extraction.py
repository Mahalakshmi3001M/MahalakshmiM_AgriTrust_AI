import os
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

train_df = pd.read_csv("train.csv")
val_df = pd.read_csv("validation.csv")
test_df = pd.read_csv("test.csv")

# --------------------------------------------------
# LOAD PRETRAINED CNN
# --------------------------------------------------

print("=" * 60)
print("       AGRITRUST AI - FEATURE EXTRACTION")
print("=" * 60)

print("\nLoading MobileNetV2...")

model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    pooling="avg"
)

model.trainable = False

print("MobileNetV2 loaded successfully.")

# --------------------------------------------------
# FEATURE EXTRACTION FUNCTION
# --------------------------------------------------

def extract_features(df):

    features = []
    labels = []

    for i, row in df.iterrows():

        image_path = row["image_path"]

        try:
            image = load_img(
                image_path,
                target_size=IMAGE_SIZE
            )

            image = img_to_array(image)
            image = np.expand_dims(image, axis=0)

            image = preprocess_input(image)

            feature = model.predict(
                image,
                verbose=0
            )[0]

            features.append(feature)
            labels.append(row["label"])

        except Exception as e:

            print("Error:", image_path)
            print(e)

        if (i + 1) % 100 == 0:
            print(f"Processed: {i + 1}/{len(df)}")

    return np.array(features), np.array(labels)


# --------------------------------------------------
# EXTRACT FEATURES
# --------------------------------------------------

print("\nExtracting training features...")
X_train, y_train = extract_features(train_df)

print("\nExtracting validation features...")
X_val, y_val = extract_features(val_df)

print("\nExtracting testing features...")
X_test, y_test = extract_features(test_df)

# --------------------------------------------------
# SAVE FEATURES
# --------------------------------------------------

np.save("X_train.npy", X_train)
np.save("y_train.npy", y_train)

np.save("X_val.npy", X_val)
np.save("y_val.npy", y_val)

np.save("X_test.npy", X_test)
np.save("y_test.npy", y_test)

print("\n" + "=" * 60)
print("FEATURE EXTRACTION COMPLETED")
print("=" * 60)

print("\nTraining feature shape   :", X_train.shape)
print("Validation feature shape :", X_val.shape)
print("Testing feature shape    :", X_test.shape)

print("\nFeature files saved successfully.")