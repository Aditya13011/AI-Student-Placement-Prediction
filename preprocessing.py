# src/preprocessing.py
# Reusable functions: load, clean, split, and build the preprocessing pipeline.

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
TARGET = "Placement"
ID_COLUMN = "College_ID"
BINARY_COLUMN = "Internship_Experience"
NUMERIC_FEATURES = [
    "IQ", "Prev_Sem_Result", "CGPA", "Academic_Performance",
    "Extra_Curricular_Score", "Communication_Skills", "Projects_Completed",
]
FEATURES = NUMERIC_FEATURES + [BINARY_COLUMN]
DATA_PATH = Path(__file__).resolve().parent / "college_student_placement_dataset.csv"

# Valid ranges come from domain knowledge (a 10-point scale), NOT from the data
VALID_RANGES = {"CGPA": (0, 10), "Prev_Sem_Result": (0, 10)}


def load_data(path=DATA_PATH):
    """Read the CSV and check that the expected columns exist."""
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")
    return df


def clean_data(df):
    """Remove duplicates, encode Yes/No, fix impossible values. Returns (clean_df, report)."""
    df = df.copy()
    report = {"rows_before": len(df)}

    # 1. Duplicates
    df = df.drop_duplicates()
    report["duplicates_removed"] = report["rows_before"] - len(df)

    # 2. Missing values (rows with no target are useless; feature gaps are imputed later)
    report["missing_feature_cells"] = int(df[FEATURES].isnull().sum().sum())
    df = df.dropna(subset=[TARGET])

    # 3. Encode Yes/No text into 1/0 (the target is encoded here too)
    yes_no = {"Yes": 1, "No": 0}
    df[BINARY_COLUMN] = df[BINARY_COLUMN].astype(str).str.strip().str.title().map(yes_no)
    df[TARGET] = df[TARGET].astype(str).str.strip().str.title().map(yes_no)
    if df[BINARY_COLUMN].isnull().any() or df[TARGET].isnull().any():
        raise ValueError("Unexpected values found in a Yes/No column")

    # 4. Impossible values: cap to the valid range and count how many were changed
    for col, (low, high) in VALID_RANGES.items():
        bad = int(((df[col] < low) | (df[col] > high)).sum())
        report[f"{col}_out_of_range_fixed"] = bad
        df[col] = df[col].clip(low, high)

    # 5. Drop the identifier column
    df = df.drop(columns=[ID_COLUMN], errors="ignore")

    report["rows_after"] = len(df)
    return df, report


def outlier_report(df, columns=NUMERIC_FEATURES):
    """Count outliers with the IQR rule. For information only; we do not delete them."""
    rows = []
    for col in columns:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_out = int(((df[col] < low) | (df[col] > high)).sum())
        rows.append({"feature": col, "lower_fence": round(low, 2),
                     "upper_fence": round(high, 2), "n_outliers": n_out})
    return pd.DataFrame(rows)


def split_data(df, test_size=0.2):
    """Stratified train-test split: both parts keep the same Placed/Not Placed ratio."""
    X = df[FEATURES]
    y = df[TARGET]
    return train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
    )


def build_preprocessor(scale=True):
    """Imputer (+ scaler) for numeric columns; the 0/1 column passes through unchanged.
    Use scale=True for Logistic Regression, KNN, SVM. Use scale=False for tree models."""
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        numeric_steps.append(("scaler", StandardScaler()))
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline(numeric_steps), NUMERIC_FEATURES),
            ("bin", "passthrough", [BINARY_COLUMN]),
        ]
    )


def feature_ranking(X_train, y_train):
    """Rank features by mutual information with the target (TRAINING data only)."""
    scores = mutual_info_classif(X_train, y_train, random_state=RANDOM_STATE)
    return pd.Series(scores, index=X_train.columns).sort_values(ascending=False).round(4)


if __name__ == "__main__":
    raw = load_data()
    print("Raw shape:", raw.shape)

    clean, report = clean_data(raw)
    print("\nCleaning report:")
    for key, value in report.items():
        print(f"  {key}: {value}")

    print("\nOutlier report (IQR rule, informational only):")
    print(outlier_report(clean).to_string(index=False))

    X_train, X_test, y_train, y_test = split_data(clean)
    print("\nTrain shape:", X_train.shape, "| Test shape:", X_test.shape)
    print(f"Placed rate -> train: {y_train.mean() * 100:.2f}% | test: {y_test.mean() * 100:.2f}%")

    print("\nFeature ranking (mutual information, train only):")
    print(feature_ranking(X_train, y_train))

    pre = build_preprocessor(scale=True)
    X_train_t = pre.fit_transform(X_train)     # learn scaling from TRAIN only
    X_test_t = pre.transform(X_test)           # apply the same scaling to TEST
    print("\nTransformed train shape:", X_train_t.shape)
    print("Scaled numeric column means on train (should be ~0):", X_train_t[:, :7].mean(axis=0).round(2))