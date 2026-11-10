import os
import pandas as pd

DATASET_PATH = "Indian_Crop_Disease_Dataset/Crop Dataset"

data = []

for crop in sorted(os.listdir(DATASET_PATH)):

    crop_path = os.path.join(DATASET_PATH, crop)

    if not os.path.isdir(crop_path):
        continue

    for condition in sorted(os.listdir(crop_path)):

        condition_path = os.path.join(crop_path, condition)

        if not os.path.isdir(condition_path):
            continue

        # Standardize condition name
        condition_clean = condition.strip().lower()

        if condition_clean == "healthy":
            condition_clean = "Healthy"
            label = 0

        else:
            condition_clean = "Disease"
            label = 1

        for root, dirs, files in os.walk(condition_path):

            for file in files:

                if file.lower().endswith(
                    (".jpg", ".jpeg", ".png", ".bmp", ".webp")
                ):

                    image_path = os.path.join(root, file)

                    data.append([
                        image_path,
                        crop,
                        condition_clean,
                        label
                    ])

df = pd.DataFrame(
    data,
    columns=["image_path", "crop", "condition", "label"]
)

print("=" * 60)
print("       AGRITRUST AI - CLEANED DATASET")
print("=" * 60)

print("\nTotal Images:", len(df))

print("\nCrop Distribution:")
print(df["crop"].value_counts())

print("\nCondition Distribution:")
print(df["condition"].value_counts())

print("\nCrop + Condition:")
print(
    df.groupby(["crop", "condition"])
      .size()
      .unstack(fill_value=0)
)

# Save cleaned metadata
df.to_csv("dataset_metadata.csv", index=False)

print("\nCleaned metadata saved successfully.")