
import pandas as pd
import json
import os

DATA = "data/uci_student_outcomes.csv"
META = "models/model_metadata.json"

df = pd.read_csv(DATA)

with open(META, "r", encoding="utf-8") as f:
    metadata = json.load(f)

target = metadata["target"]

print("\n========== DATASET ANALYSIS ==========")

print("Dataset shape:", df.shape)
print("\nMissing values:")
print(df.isnull().sum().sum())

print("\nDuplicate rows:")
print(df.duplicated().sum())

print("\nClass distribution:")
print(df[target].value_counts())

print("\nClass percentages:")
print(
    (df[target].value_counts(normalize=True) * 100).round(2)
)

print("\n========== MODEL PERFORMANCE ==========")

print("Test accuracy:",
      round(metadata["accuracy"] * 100, 2), "%")

print("CV accuracy:",
      round(metadata.get("cv_accuracy", 0) * 100, 2), "%")

print("\nClass-wise report:")

for cls, values in metadata["report"].items():
    if isinstance(values, dict) and "recall" in values:
        print(
            cls,
            "Precision:", round(values["precision"], 3),
            "Recall:", round(values["recall"], 3),
            "F1:", round(values["f1-score"], 3)
        )

print("\nConfusion matrix:")
print(metadata["confusion_matrix"])

print("\n========== FEATURE INFORMATION ==========")

print("Number of features:", len(metadata["features"]))

print("Features:")
for feature in metadata["features"]:
    print("-", feature)
