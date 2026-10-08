import pandas as pd

df = pd.read_csv("college_student_placement_dataset.csv")

print("=" * 70)
print("DATASET DIAGNOSTIC")
print("=" * 70)

# ------------------------------------------------------------
# 1. Dataset information
# ------------------------------------------------------------

print("\nDATASET SHAPE:")
print(df.shape)

print("\nDUPLICATE ROWS:")
print(df.duplicated().sum())


# ------------------------------------------------------------
# 2. Target distribution
# ------------------------------------------------------------

print("\nPLACEMENT DISTRIBUTION:")
print(df["Placement"].value_counts())

print("\nPLACEMENT PERCENTAGE:")
print(
    (df["Placement"].value_counts(normalize=True) * 100)
    .round(2)
)


# ------------------------------------------------------------
# 3. Unique values
# ------------------------------------------------------------

print("\nUNIQUE VALUES PER COLUMN:")

for column in df.columns:
    print(
        f"{column:30s} : "
        f"{df[column].nunique()} unique values"
    )


# ------------------------------------------------------------
# 4. Average numerical values by placement
# ------------------------------------------------------------

numeric_columns = [
    "IQ",
    "Prev_Sem_Result",
    "CGPA",
    "Academic_Performance",
    "Extra_Curricular_Score",
    "Communication_Skills",
    "Projects_Completed"
]

print("\nAVERAGE VALUES BY PLACEMENT:")

print(
    df.groupby("Placement")[numeric_columns]
      .mean()
      .round(3)
)


# ------------------------------------------------------------
# 5. Minimum and maximum values by placement
# ------------------------------------------------------------

print("\nMINIMUM VALUES BY PLACEMENT:")

print(
    df.groupby("Placement")[numeric_columns]
      .min()
      .round(3)
)

print("\nMAXIMUM VALUES BY PLACEMENT:")

print(
    df.groupby("Placement")[numeric_columns]
      .max()
      .round(3)
)


# ------------------------------------------------------------
# 6. Internship vs placement
# ------------------------------------------------------------

print("\nINTERNSHIP VS PLACEMENT:")

print(
    pd.crosstab(
        df["Internship_Experience"],
        df["Placement"],
        margins=True
    )
)


print("\nINTERNSHIP VS PLACEMENT (%):")

print(
    pd.crosstab(
        df["Internship_Experience"],
        df["Placement"],
        normalize="index"
    ).round(3) * 100
)


# ------------------------------------------------------------
# 7. Look for suspiciously perfect separation
# ------------------------------------------------------------

print("\nPLACEMENT BY CGPA:")

cgpa_groups = pd.cut(
    df["CGPA"],
    bins=[0, 5, 6, 7, 8, 9, 10],
    include_lowest=True
)

print(
    pd.crosstab(
        cgpa_groups,
        df["Placement"],
        normalize="index"
    ).round(3) * 100
)


# ------------------------------------------------------------
# 8. Correlation with binary placement
# ------------------------------------------------------------

df["Placement_Binary"] = df["Placement"].map({
    "No": 0,
    "Yes": 1
})

correlations = (
    df[numeric_columns + ["Placement_Binary"]]
    .corr()["Placement_Binary"]
    .drop("Placement_Binary")
    .sort_values(
        ascending=False
    )
)

print("\nCORRELATION WITH PLACEMENT:")

print(correlations.round(4))


print("\n" + "=" * 70)
print("DIAGNOSTIC COMPLETED")
print("=" * 70)