import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. LOAD DATASET
# ============================================================

df = pd.read_csv("college_student_placement_dataset.csv")

print("=" * 60)
print("AI-BASED STUDENT PLACEMENT PREDICTION")
print("EXPLORATORY DATA ANALYSIS")
print("=" * 60)


# ============================================================
# 2. BASIC INFORMATION
# ============================================================

print("\n--- DATASET SHAPE ---")
print(df.shape)

print("\n--- COLUMN NAMES ---")
print(df.columns.tolist())

print("\n--- DATA TYPES ---")
print(df.dtypes)

print("\n--- FIRST 5 RECORDS ---")
print(df.head())


# ============================================================
# 3. MISSING VALUES
# ============================================================

print("\n--- MISSING VALUES ---")
print(df.isnull().sum())

print("\nTotal missing values:")
print(df.isnull().sum().sum())


# ============================================================
# 4. DUPLICATE RECORDS
# ============================================================

print("\n--- DUPLICATE RECORDS ---")
print("Number of duplicate rows:", df.duplicated().sum())


# ============================================================
# 5. PLACEMENT DISTRIBUTION
# ============================================================

print("\n--- PLACEMENT DISTRIBUTION ---")
print(df["Placement"].value_counts())

print("\n--- PLACEMENT PERCENTAGE ---")
print(df["Placement"].value_counts(normalize=True) * 100)


# ============================================================
# 6. STATISTICAL SUMMARY
# ============================================================

print("\n--- STATISTICAL SUMMARY ---")
print(df.describe())


# ============================================================
# 7. UNIQUE VALUES OF CATEGORICAL COLUMNS
# ============================================================

print("\n--- UNIQUE VALUES ---")

print("\nPlacement:")
print(df["Placement"].unique())

print("\nInternship Experience:")
print(df["Internship_Experience"].unique())

print("\nCollege IDs:")
print("Number of unique College IDs:", df["College_ID"].nunique())


# ============================================================
# 8. PLACEMENT COUNT GRAPH
# ============================================================

plt.figure(figsize=(7, 5))

df["Placement"].value_counts().plot(kind="bar")

plt.title("Student Placement Distribution")
plt.xlabel("Placement Status")
plt.ylabel("Number of Students")
plt.xticks(rotation=0)

plt.tight_layout()
plt.savefig("placement_distribution.png")
plt.show()


# ============================================================
# 9. CGPA VS PLACEMENT
# ============================================================

plt.figure(figsize=(7, 5))

df.boxplot(
    column="CGPA",
    by="Placement"
)

plt.title("CGPA Distribution by Placement Status")
plt.suptitle("")
plt.xlabel("Placement Status")
plt.ylabel("CGPA")

plt.tight_layout()
plt.savefig("cgpa_vs_placement.png")
plt.show()


# ============================================================
# 10. IQ VS PLACEMENT
# ============================================================

plt.figure(figsize=(7, 5))

df.boxplot(
    column="IQ",
    by="Placement"
)

plt.title("IQ Distribution by Placement Status")
plt.suptitle("")
plt.xlabel("Placement Status")
plt.ylabel("IQ")

plt.tight_layout()
plt.savefig("iq_vs_placement.png")
plt.show()


# ============================================================
# 11. PREVIOUS SEMESTER RESULT VS PLACEMENT
# ============================================================

plt.figure(figsize=(7, 5))

df.boxplot(
    column="Prev_Sem_Result",
    by="Placement"
)

plt.title("Previous Semester Result vs Placement")
plt.suptitle("")
plt.xlabel("Placement Status")
plt.ylabel("Previous Semester Result")

plt.tight_layout()
plt.savefig("previous_sem_result.png")
plt.show()


# ============================================================
# 12. ACADEMIC PERFORMANCE VS PLACEMENT
# ============================================================

plt.figure(figsize=(7, 5))

df.boxplot(
    column="Academic_Performance",
    by="Placement"
)

plt.title("Academic Performance vs Placement")
plt.suptitle("")
plt.xlabel("Placement Status")
plt.ylabel("Academic Performance")

plt.tight_layout()
plt.savefig("academic_performance.png")
plt.show()


# ============================================================
# 13. COMMUNICATION SKILLS VS PLACEMENT
# ============================================================

plt.figure(figsize=(7, 5))

df.boxplot(
    column="Communication_Skills",
    by="Placement"
)

plt.title("Communication Skills vs Placement")
plt.suptitle("")
plt.xlabel("Placement Status")
plt.ylabel("Communication Skills")

plt.tight_layout()
plt.savefig("communication_skills.png")
plt.show()


# ============================================================
# 14. PROJECTS COMPLETED VS PLACEMENT
# ============================================================

plt.figure(figsize=(7, 5))

df.boxplot(
    column="Projects_Completed",
    by="Placement"
)

plt.title("Projects Completed vs Placement")
plt.suptitle("")
plt.xlabel("Placement Status")
plt.ylabel("Number of Projects")

plt.tight_layout()
plt.savefig("projects_vs_placement.png")
plt.show()


# ============================================================
# 15. INTERNSHIP EXPERIENCE VS PLACEMENT
# ============================================================

print("\n--- INTERNSHIP EXPERIENCE VS PLACEMENT ---")

internship_table = pd.crosstab(
    df["Internship_Experience"],
    df["Placement"],
    normalize="index"
) * 100

print(internship_table)


# ============================================================
# 16. AVERAGE FEATURES BY PLACEMENT
# ============================================================

numeric_features = [
    "IQ",
    "Prev_Sem_Result",
    "CGPA",
    "Academic_Performance",
    "Extra_Curricular_Score",
    "Communication_Skills",
    "Projects_Completed"
]

print("\n--- AVERAGE FEATURE VALUES BY PLACEMENT ---")

print(
    df.groupby("Placement")[numeric_features].mean().round(2)
)


# ============================================================
# 17. CORRELATION MATRIX
# ============================================================

print("\n--- CORRELATION MATRIX ---")

correlation = df[numeric_features].corr()

print(correlation.round(2))

plt.figure(figsize=(10, 7))

plt.imshow(correlation, cmap="coolwarm", interpolation="nearest")

plt.colorbar()

plt.xticks(
    range(len(correlation.columns)),
    correlation.columns,
    rotation=45,
    ha="right"
)

plt.yticks(
    range(len(correlation.columns)),
    correlation.columns
)

plt.title("Feature Correlation Matrix")

plt.tight_layout()
plt.savefig("correlation_matrix.png")
plt.show()


# ============================================================
# 18. SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("EDA COMPLETED SUCCESSFULLY")
print("=" * 60)

print("\nGenerated graphs:")
print("1. placement_distribution.png")
print("2. cgpa_vs_placement.png")
print("3. iq_vs_placement.png")
print("4. previous_sem_result.png")
print("5. academic_performance.png")
print("6. communication_skills.png")
print("7. projects_vs_placement.png")
print("8. correlation_matrix.png")