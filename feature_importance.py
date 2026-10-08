import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


# ============================================================
# 1. LOAD DATASET
# ============================================================

df = pd.read_csv(
    "college_student_placement_dataset.csv"
)

print("=" * 70)
print("RANDOM FOREST FEATURE IMPORTANCE ANALYSIS")
print("=" * 70)


# ============================================================
# 2. PREPARE DATA
# ============================================================

# Remove identifier
df = df.drop(
    columns=["College_ID"]
)

# Convert Internship Experience
df["Internship_Experience"] = df[
    "Internship_Experience"
].map({
    "Yes": 1,
    "No": 0
})

# Convert target
df["Placement"] = df[
    "Placement"
].map({
    "Yes": 1,
    "No": 0
})


# ============================================================
# 3. HANDLE MISSING VALUES
# ============================================================

numeric_columns = df.select_dtypes(
    include=["int64", "float64"]
).columns

for column in numeric_columns:

    df[column] = df[column].fillna(
        df[column].median()
    )


# ============================================================
# 4. FEATURES AND TARGET
# ============================================================

X = df.drop(
    columns=["Placement"]
)

y = df["Placement"]


# ============================================================
# 5. TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42,

    stratify=y
)


# ============================================================
# 6. TRAIN FINAL RANDOM FOREST
# ============================================================

model = RandomForestClassifier(

    n_estimators=200,

    max_depth=8,

    min_samples_leaf=10,

    class_weight="balanced",

    random_state=42,

    n_jobs=-1
)


model.fit(
    X_train,
    y_train
)


# ============================================================
# 7. CALCULATE FEATURE IMPORTANCE
# ============================================================

importance_df = pd.DataFrame({

    "Feature": X.columns,

    "Importance": model.feature_importances_

})


importance_df = importance_df.sort_values(

    by="Importance",

    ascending=False
)


# ============================================================
# 8. DISPLAY RESULTS
# ============================================================

print("\nFEATURE IMPORTANCE:")

print(
    importance_df.to_string(
        index=False,
        float_format=lambda x:
        f"{x:.4f}"
    )
)


# ============================================================
# 9. SAVE RESULTS
# ============================================================

importance_df.to_csv(

    "feature_importance.csv",

    index=False
)


# ============================================================
# 10. CREATE GRAPH
# ============================================================

plt.figure(
    figsize=(10, 6)
)

plt.barh(

    importance_df["Feature"],

    importance_df["Importance"]
)

plt.xlabel(
    "Importance"
)

plt.ylabel(
    "Feature"
)

plt.title(
    "Random Forest Feature Importance - Student Placement"
)

plt.gca().invert_yaxis()

plt.tight_layout()


plt.savefig(

    "feature_importance.png",

    dpi=300
)


plt.show()


# ============================================================
# 11. FINISHED
# ============================================================

print("\nFeature importance saved as:")

print("feature_importance.csv")

print("\nGraph saved as:")

print("feature_importance.png")

print("\n" + "=" * 70)

print("FEATURE IMPORTANCE ANALYSIS COMPLETED")

print("=" * 70)