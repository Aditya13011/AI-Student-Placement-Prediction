# src/profile_data.py
# Detailed profile of the Kaggle placement CSV.
# Usage: python profile_data.py [path/to/dataset.csv]

import argparse
from pathlib import Path
import sys
import pandas as pd

parser = argparse.ArgumentParser(description="Profile the student placement dataset.")
parser.add_argument(
    "csv_path",
    nargs="?",
    default=Path(__file__).with_name("college_student_placement_dataset.csv"),
    help="CSV file to profile (defaults to the dataset in this folder)",
)
args = parser.parse_args()

df = pd.read_csv(args.csv_path)
target = "Placement"

print("Shape:", df.shape)
print("\nColumns and types:\n", df.dtypes)
print("\nMissing values:\n", df.isnull().sum())
print("\nDuplicate rows:", df.duplicated().sum())

print("\nTarget distribution (%):")
print((df[target].value_counts(normalize=True) * 100).round(2))

print("\nNumeric summary:")
print(df.describe().T)

print("\nCategorical columns:")
for col in df.select_dtypes(include="object").columns:
    print(f"  {col}: {df[col].unique()[:10]}")

print("\nUnique values per column:")
print(df.nunique())

# Correlation of every numeric feature with the target (Yes=1, No=0)
tmp = df.copy()
tmp[target] = (tmp[target] == "Yes").astype(int)
tmp = pd.get_dummies(tmp, drop_first=True, dtype=int)
print("\nCorrelation with Placement:")
print(tmp.corr(numeric_only=True)["Placement"].drop("Placement").sort_values())