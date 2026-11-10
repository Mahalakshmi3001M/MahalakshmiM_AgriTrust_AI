import os

DATASET_PATH = "Indian_Crop_Disease_Dataset/Crop Dataset"

print("=" * 60)
print("       AGRITRUST AI - DISEASE CLASS INSPECTION")
print("=" * 60)

total_images = 0
total_classes = 0

for crop in sorted(os.listdir(DATASET_PATH)):

    crop_path = os.path.join(DATASET_PATH, crop)

    if not os.path.isdir(crop_path):
        continue

    print(f"\n🌱 CROP: {crop.upper()}")
    print("-" * 40)

    disease_found = False

    for item in sorted(os.listdir(crop_path)):

        item_path = os.path.join(crop_path, item)

        if os.path.isdir(item_path):

            image_count = 0

            for root, dirs, files in os.walk(item_path):
                for file in files:

                    if file.lower().endswith(
                        (".jpg", ".jpeg", ".png", ".bmp", ".webp")
                    ):
                        image_count += 1

            print(f"{item:<30} : {image_count} images")

            total_images += image_count
            total_classes += 1
            disease_found = True

    if not disease_found:
        print("No disease subfolders found.")

print("\n" + "=" * 60)
print("TOTAL DISEASE/HEALTH CLASSES:", total_classes)
print("TOTAL IMAGES:", total_images)
print("=" * 60)