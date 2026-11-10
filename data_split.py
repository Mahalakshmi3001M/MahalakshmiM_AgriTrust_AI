import pandas as pd
from sklearn.model_selection import train_test_split

# Load cleaned metadata
df = pd.read_csv("dataset_metadata.csv")

print("=" * 60)
print("       AGRITRUST AI - DATASET SPLITTING")
print("=" * 60)

# First split: 70% training, 30% temporary
train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    stratify=df["condition"],
    random_state=42
)

# Second split: 15% validation, 15% testing
val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    stratify=temp_df["condition"],
    random_state=42
)

print("\nDataset Sizes:")
print("Total      :", len(df))
print("Training   :", len(train_df))
print("Validation :", len(val_df))
print("Testing    :", len(test_df))

print("\nTraining Distribution:")
print(train_df["condition"].value_counts())

print("\nValidation Distribution:")
print(val_df["condition"].value_counts())

print("\nTesting Distribution:")
print(test_df["condition"].value_counts())

# Save split files
train_df.to_csv("train.csv", index=False)
val_df.to_csv("validation.csv", index=False)
test_df.to_csv("test.csv", index=False)

print("\nSplit files saved successfully:")
print("train.csv")
print("validation.csv")
print("test.csv")