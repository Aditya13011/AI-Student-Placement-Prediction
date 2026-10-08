import pandas as pd
import numpy as np

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif


# ============================================================
# MODEL VALIDATION / DATA LEAKAGE ANALYSIS
# ============================================================

print("=" * 70)
print("MODEL VALIDATION AND DATA LEAKAGE ANALYSIS")
print("=" * 70)


# ============================================================
# 1. LOAD DATASET
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "college_student_placement_dataset.csv"

df = pd.read_csv(DATA_PATH)

print("\nDATASET LOADED")
print("-" * 70)

print(f"Dataset shape: {df.shape}")
print(f"Rows: {df.shape[0]}")
print(f"Columns: {df.shape[1]}")


# ============================================================
# 2. BASIC INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("1. BASIC DATASET INFORMATION")
print("=" * 70)

print("\nColumns:")

for column in df.columns:
    print(f"- {column}")


print("\nData types:")

print(df.dtypes)


# ============================================================
# 3. TARGET DISTRIBUTION
# ============================================================

TARGET = "Placement"

if TARGET not in df.columns:

    raise ValueError(
        f"Target column '{TARGET}' was not found."
    )


print("\n" + "=" * 70)
print("2. TARGET DISTRIBUTION")
print("=" * 70)

print(df[TARGET].value_counts())

print("\nTarget percentage:")

print(
    df[TARGET]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)


# ============================================================
# 4. DUPLICATE ROW CHECK
# ============================================================

print("\n" + "=" * 70)
print("3. DUPLICATE ROW CHECK")
print("=" * 70)

duplicate_rows = df.duplicated().sum()

print(
    f"Duplicate complete rows: {duplicate_rows}"
)

if duplicate_rows == 0:

    print(
        "✓ No duplicate complete rows found."
    )

else:

    print(
        "⚠ Duplicate rows found."
    )


# ============================================================
# 5. MISSING VALUE CHECK
# ============================================================

print("\n" + "=" * 70)
print("4. MISSING VALUE CHECK")
print("=" * 70)

missing_values = df.isnull().sum()

print(missing_values)

if missing_values.sum() == 0:

    print(
        "\n✓ No missing values found."
    )

else:

    print(
        "\n⚠ Missing values exist."
    )


# ============================================================
# 6. SEPARATE FEATURES AND TARGET
# ============================================================

X = df.drop(
    columns=[TARGET]
).copy()

y = df[TARGET].copy()


# ============================================================
# 7. TARGET ENCODING
# ============================================================

print("\n" + "=" * 70)
print("5. TARGET ENCODING")
print("=" * 70)

print("Original target values:")

print(y.unique())


# Convert target to ordinary Python/string values first.
# This handles pandas ArrowStringArray correctly.

y_string = y.astype("string").astype(str)


# Explicit handling for Yes / No

unique_targets = set(
    value.strip().lower()
    for value in y_string.unique()
)


if unique_targets == {"yes", "no"}:

    y_encoded = (
        y_string
        .str.strip()
        .str.lower()
        .map(
            {
                "no": 0,
                "yes": 1
            }
        )
        .astype(int)
        .to_numpy()
    )

    print("\nTarget mapping:")

    print("No  -> 0")
    print("Yes -> 1")


else:

    # General fallback for other target formats

    target_encoder = LabelEncoder()

    y_encoded = target_encoder.fit_transform(
        y_string
    )

    print("\nEncoded target mapping:")

    for index, class_name in enumerate(
        target_encoder.classes_
    ):

        print(
            f"{class_name} -> {index}"
        )


# ============================================================
# 8. ENCODE CATEGORICAL FEATURES
# ============================================================

X_encoded = X.copy()

categorical_columns = X_encoded.select_dtypes(
    include=["object", "category", "string"]
).columns.tolist()


print("\n" + "=" * 70)
print("6. FEATURE ENCODING")
print("=" * 70)


if len(categorical_columns) == 0:

    print(
        "No categorical feature columns found."
    )

else:

    print(
        "Categorical columns:"
    )

    for column in categorical_columns:

        print(
            f"- {column}"
        )

        encoder = LabelEncoder()

        X_encoded[column] = encoder.fit_transform(
            X_encoded[column].astype(str)
        )


# ============================================================
# 9. CONVERT FEATURES TO NUMERIC
# ============================================================

for column in X_encoded.columns:

    X_encoded[column] = pd.to_numeric(
        X_encoded[column],
        errors="coerce"
    )


# Check for conversion problems

conversion_missing = (
    X_encoded.isnull().sum()
)


if conversion_missing.sum() > 0:

    print(
        "\n⚠ Some feature values could not be "
        "converted to numeric."
    )

    print(
        conversion_missing[
            conversion_missing > 0
        ]
    )

    raise ValueError(
        "Feature conversion created missing values. "
        "Please inspect the dataset."
    )


# ============================================================
# 10. FEATURE-TARGET CORRELATION
# ============================================================

print("\n" + "=" * 70)
print("7. FEATURE-TARGET CORRELATION")
print("=" * 70)


correlation_df = X_encoded.copy()

correlation_df["Placement_Binary"] = y_encoded


correlations = (
    correlation_df
    .corr(numeric_only=True)["Placement_Binary"]
    .drop("Placement_Binary")
    .sort_values(
        key=abs,
        ascending=False
    )
)


print(
    "\nCorrelation with Placement:"
)


for feature, value in correlations.items():

    print(
        f"{feature:<30} {value:.4f}"
    )


# ============================================================
# 11. HIGH CORRELATION FEATURES
# ============================================================

print("\n" + "=" * 70)
print("8. HIGH-CORRELATION FEATURES")
print("=" * 70)


high_correlation_features = []


for feature, value in correlations.items():

    if abs(value) >= 0.80:

        high_correlation_features.append(
            feature
        )

        print(
            f"⚠ {feature}: "
            f"correlation = {value:.4f}"
        )


if len(high_correlation_features) == 0:

    print(
        "✓ No feature has correlation >= 0.80."
    )

else:

    print(
        "\nThese features require investigation."
    )


# ============================================================
# 12. SINGLE-FEATURE DECISION TREE TEST
# ============================================================

print("\n" + "=" * 70)
print("9. SINGLE-FEATURE PREDICTION TEST")
print("=" * 70)


print(
    "\nTesting whether any single feature can "
    "almost perfectly predict Placement."
)


single_feature_results = []


for feature in X_encoded.columns:

    X_feature = X_encoded[
        [feature]
    ]


    X_train, X_test, y_train, y_test = train_test_split(
        X_feature,
        y_encoded,
        test_size=0.20,
        random_state=42,
        stratify=y_encoded
    )


    model = DecisionTreeClassifier(
        max_depth=3,
        random_state=42
    )


    model.fit(
        X_train,
        y_train
    )


    predictions = model.predict(
        X_test
    )


    accuracy = accuracy_score(
        y_test,
        predictions
    )


    single_feature_results.append(
        {
            "Feature": feature,
            "Accuracy": accuracy
        }
    )


single_feature_df = pd.DataFrame(
    single_feature_results
).sort_values(
    "Accuracy",
    ascending=False
)


print(
    "\nSingle-feature results:"
)


print(
    single_feature_df.to_string(
        index=False
    )
)


print(
    "\nPotentially suspicious single features:"
)


suspicious_single_features = (
    single_feature_df[
        single_feature_df["Accuracy"] >= 0.95
    ]
)


if len(suspicious_single_features) == 0:

    print(
        "✓ No single feature achieved >= 95% accuracy."
    )

else:

    print(
        suspicious_single_features.to_string(
            index=False
        )
    )


# ============================================================
# 13. FEATURE VALUE -> TARGET CONSISTENCY
# ============================================================

print("\n" + "=" * 70)
print("10. FEATURE VALUE -> TARGET CONSISTENCY CHECK")
print("=" * 70)


consistency_results = []


for feature in X.columns:

    grouped = (
        df.groupby(feature)[TARGET]
        .nunique()
    )


    total_unique_values = len(
        grouped
    )


    perfectly_consistent_values = (
        grouped == 1
    ).sum()


    if total_unique_values > 0:

        consistency_percentage = (
            perfectly_consistent_values
            / total_unique_values
        ) * 100

    else:

        consistency_percentage = 0


    consistency_results.append(
        {
            "Feature": feature,
            "Unique_Values": df[feature].nunique(),
            "Consistent_Target_Values":
                perfectly_consistent_values,
            "Consistency_Percentage":
                consistency_percentage
        }
    )


consistency_df = pd.DataFrame(
    consistency_results
).sort_values(
    "Consistency_Percentage",
    ascending=False
)


print(
    consistency_df.to_string(
        index=False
    )
)


# ============================================================
# 14. MUTUAL INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("11. MUTUAL INFORMATION ANALYSIS")
print("=" * 70)


try:

    mutual_information = (
        mutual_info_classif(
            X_encoded,
            y_encoded,
            random_state=42
        )
    )


    mi_df = pd.DataFrame(
        {
            "Feature": X_encoded.columns,
            "Mutual_Information":
                mutual_information
        }
    ).sort_values(
        "Mutual_Information",
        ascending=False
    )


    print(
        mi_df.to_string(
            index=False
        )
    )


except Exception as e:

    print(
        f"Mutual information calculation failed: {e}"
    )

    mi_df = pd.DataFrame()


# ============================================================
# 15. FRESH RANDOM FOREST VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("12. FRESH RANDOM FOREST VALIDATION")
print("=" * 70)


X_train, X_test, y_train, y_test = train_test_split(
    X_encoded,
    y_encoded,
    test_size=0.20,
    random_state=42,
    stratify=y_encoded
)


rf_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)


print(
    "\nTraining fresh Random Forest..."
)


rf_model.fit(
    X_train,
    y_train
)


predictions = rf_model.predict(
    X_test
)


probabilities = rf_model.predict_proba(
    X_test
)[:, 1]


accuracy = accuracy_score(
    y_test,
    predictions
)


precision = precision_score(
    y_test,
    predictions,
    zero_division=0
)


recall = recall_score(
    y_test,
    predictions,
    zero_division=0
)


f1 = f1_score(
    y_test,
    predictions,
    zero_division=0
)


roc_auc = roc_auc_score(
    y_test,
    probabilities
)


print(
    "\nFRESH RANDOM FOREST RESULTS"
)

print("-" * 50)

print(
    f"Accuracy : {accuracy:.4f}"
)

print(
    f"Precision: {precision:.4f}"
)

print(
    f"Recall   : {recall:.4f}"
)

print(
    f"F1 Score : {f1:.4f}"
)

print(
    f"ROC-AUC  : {roc_auc:.4f}"
)


# ============================================================
# 16. FEATURE IMPORTANCE
# ============================================================

print("\n" + "=" * 70)
print("13. RANDOM FOREST FEATURE IMPORTANCE")
print("=" * 70)


importance_df = pd.DataFrame(
    {
        "Feature": X_encoded.columns,
        "Importance":
            rf_model.feature_importances_
    }
).sort_values(
    "Importance",
    ascending=False
)


print(
    importance_df.to_string(
        index=False
    )
)


# ============================================================
# 17. TRAIN / TEST OVERLAP
# ============================================================

print("\n" + "=" * 70)
print("14. TRAIN / TEST OVERLAP CHECK")
print("=" * 70)


train_indices = set(
    X_train.index
)

test_indices = set(
    X_test.index
)


overlap = (
    train_indices
    .intersection(test_indices)
)


print(
    f"Training samples: {len(train_indices)}"
)

print(
    f"Testing samples : {len(test_indices)}"
)

print(
    f"Overlapping rows: {len(overlap)}"
)


if len(overlap) == 0:

    print(
        "✓ No row-level train/test overlap."
    )

else:

    print(
        "⚠ Train/test overlap detected."
    )


# ============================================================
# 18. IDENTICAL FEATURE ROW CHECK
# ============================================================

print("\n" + "=" * 70)
print("15. IDENTICAL FEATURE ROW ANALYSIS")
print("=" * 70)


feature_columns = list(
    X.columns
)


duplicate_feature_rows = df.duplicated(
    subset=feature_columns,
    keep=False
)


duplicate_feature_count = (
    duplicate_feature_rows.sum()
)


print(
    "Rows sharing identical feature values: "
    f"{duplicate_feature_count}"
)


if duplicate_feature_count == 0:

    print(
        "✓ No identical feature rows found."
    )

else:

    duplicate_feature_data = df[
        duplicate_feature_rows
    ]


    target_counts = (
        duplicate_feature_data
        .groupby(feature_columns)[TARGET]
        .nunique()
    )


    conflicting_groups = (
        target_counts[
            target_counts > 1
        ]
    )


    print(
        "Conflicting duplicate feature groups: "
        f"{len(conflicting_groups)}"
    )


    if len(conflicting_groups) == 0:

        print(
            "⚠ Identical feature rows always have "
            "the same target."
        )

        print(
            "This may contribute to very high "
            "model accuracy."
        )

    else:

        print(
            "⚠ Some identical feature rows have "
            "different target values."
        )


# ============================================================
# 19. FINAL DIAGNOSTIC SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FINAL DIAGNOSTIC SUMMARY")
print("=" * 70)


print(
    f"\nDataset size           : {df.shape[0]} rows"
)

print(
    f"Number of features     : {X.shape[1]}"
)

print(
    f"Duplicate rows         : {duplicate_rows}"
)

print(
    f"Train/Test overlap     : {len(overlap)}"
)

print(
    f"Random Forest accuracy : {accuracy * 100:.2f}%"
)

print(
    f"Random Forest F1       : {f1:.4f}"
)

print(
    f"Random Forest ROC-AUC  : {roc_auc:.4f}"
)


if len(suspicious_single_features) > 0:

    print(
        "\n⚠ WARNING:"
    )

    print(
        "One or more individual features achieved "
        "very high predictive accuracy."
    )

    print(
        "Investigate these features for possible "
        "target leakage or synthetic-data patterns."
    )

else:

    print(
        "\n✓ No single feature showed extreme "
        "predictive power."
    )


if len(high_correlation_features) > 0:

    print(
        "\n⚠ HIGH CORRELATION WARNING:"
    )

    for feature in high_correlation_features:

        print(
            f"- {feature}"
        )

else:

    print(
        "\n✓ No extremely high feature-target "
        "correlation detected."
    )


# ============================================================
# 20. SAVE SUMMARY RESULTS
# ============================================================

results_path = (
    BASE_DIR /
    "model_validation_results.csv"
)


summary_results = pd.DataFrame(
    {
        "Metric": [
            "Dataset Rows",
            "Dataset Columns",
            "Duplicate Rows",
            "Train Test Overlap",
            "Random Forest Accuracy",
            "Random Forest Precision",
            "Random Forest Recall",
            "Random Forest F1",
            "Random Forest ROC AUC"
        ],

        "Value": [
            df.shape[0],
            df.shape[1],
            duplicate_rows,
            len(overlap),
            accuracy,
            precision,
            recall,
            f1,
            roc_auc
        ]
    }
)


summary_results.to_csv(
    results_path,
    index=False
)


# ============================================================
# 21. SAVE FEATURE ANALYSIS
# ============================================================

feature_analysis_path = (
    BASE_DIR /
    "feature_validation_results.csv"
)


feature_analysis = importance_df.merge(
    correlations.rename(
        "Correlation_With_Placement"
    ),
    left_on="Feature",
    right_index=True,
    how="left"
)


feature_analysis = feature_analysis.merge(
    single_feature_df,
    on="Feature",
    how="left"
)


feature_analysis.to_csv(
    feature_analysis_path,
    index=False
)


# ============================================================
# COMPLETION
# ============================================================

print("\n" + "=" * 70)
print("RESULTS SAVED")
print("=" * 70)


print(
    f"\nCreated:"
)

print(
    f"- {results_path.name}"
)

print(
    f"- {feature_analysis_path.name}"
)


print("\n" + "=" * 70)
print("MODEL VALIDATION COMPLETED")
print("=" * 70)
