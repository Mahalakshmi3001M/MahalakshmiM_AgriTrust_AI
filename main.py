import os
from PIL import Image

DATASET_PATH = "Indian_Crop_Disease_Dataset/Crop Dataset"

print("=" * 50)
print("      AGRITRUST AI - DATASET INSPECTION")
print("=" * 50)

total_images = 0
crop_count = {}

for crop in os.listdir(DATASET_PATH):

    crop_path = os.path.join(DATASET_PATH, crop)

    if not os.path.isdir(crop_path):
        continue

    count = 0

    for root, dirs, files in os.walk(crop_path):
        for file in files:

            if file.lower().endswith(
                (".jpg", ".jpeg", ".png", ".bmp", ".webp")
            ):
                count += 1

    crop_count[crop] = count
    total_images += count

print("\nNumber of Crops:", len(crop_count))

print("\nImages in Each Crop:")
for crop, count in crop_count.items():
    print(f"{crop:<15} : {count}")

print("\nTotal Images:", total_images)

print("\nDataset Path:")
print(DATASET_PATH)