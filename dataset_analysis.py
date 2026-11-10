import os
import matplotlib.pyplot as plt

DATASET_PATH = "Indian_Crop_Disease_Dataset/Crop Dataset"

crops = []
disease_counts = []
healthy_counts = []

for crop in sorted(os.listdir(DATASET_PATH)):

    crop_path = os.path.join(DATASET_PATH, crop)

    if not os.path.isdir(crop_path):
        continue

    disease = 0
    healthy = 0

    for class_name in os.listdir(crop_path):

        class_path = os.path.join(crop_path, class_name)

        if not os.path.isdir(class_path):
            continue

        count = 0

        for root, dirs, files in os.walk(class_path):
            for file in files:
                if file.lower().endswith(
                    (".jpg", ".jpeg", ".png", ".bmp", ".webp")
                ):
                    count += 1

        if class_name.lower() == "healthy":
            healthy = count
        else:
            disease = count

    crops.append(crop)
    disease_counts.append(disease)
    healthy_counts.append(healthy)

# Display counts
print("\nCROP-WISE DATA DISTRIBUTION")
print("-" * 45)

for i in range(len(crops)):
    print(
        f"{crops[i]:<12} "
        f"Disease: {disease_counts[i]:<5} "
        f"Healthy: {healthy_counts[i]}"
    )

# Plot
x = range(len(crops))

plt.figure(figsize=(10, 6))

plt.bar(x, disease_counts, label="Disease")
plt.bar(x, healthy_counts, bottom=disease_counts, label="Healthy")

plt.xticks(x, crops)
plt.xlabel("Crop")
plt.ylabel("Number of Images")
plt.title("Crop Disease Dataset Distribution")
plt.legend()

plt.tight_layout()
plt.show()