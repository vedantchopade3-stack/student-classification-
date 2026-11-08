
import pandas as pd

file_path = "data/ecotec_online_student_risk.csv"

df = pd.read_csv(file_path)

print("\n========== DATASET INFORMATION ==========")
print("Total records:", df.shape[0])
print("Total columns:", df.shape[1])

print("\n========== COLUMN NAMES ==========")
print(df.columns.tolist())

print("\n========== DATA TYPES ==========")
print(df.dtypes)

print("\n========== FIRST 5 RECORDS ==========")
print(df.head())

print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

if "TARGET_RISK" in df.columns:
    print("\n========== TARGET DISTRIBUTION ==========")
    print(df["TARGET_RISK"].value_counts(dropna=False))
else:
    print("\nWARNING: TARGET_RISK column not found.")
