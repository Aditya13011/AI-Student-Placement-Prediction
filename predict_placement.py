import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


# ============================================================
# 1. LOAD DATASET
# ============================================================

DATA_FILE = "college_student_placement_dataset.csv"

print("=" * 70)
print("AI-BASED STUDENT PLACEMENT PREDICTION SYSTEM")
print("=" * 70)

print("\nLoading dataset...")

df = pd.read_csv(DATA_FILE)

print("Dataset loaded successfully.")
print("Dataset shape:", df.shape)


# ============================================================
# 2. DATA PREPROCESSING
# ============================================================

# Remove ID column because it has no predictive meaning
df = df.drop(columns=["College_ID"])


# Convert Internship Experience
df["Internship_Experience"] = df[
    "Internship_Experience"
].map({
    "Yes": 1,
    "No": 0
})


# Convert Placement target
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


print("\nFeatures used by the model:")

for feature in X.columns:
    print("-", feature)


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


print("\nTraining samples:", len(X_train))
print("Testing samples :", len(X_test))


# ============================================================
# 6. CREATE RANDOM FOREST MODEL
# ============================================================

model = RandomForestClassifier(

    n_estimators=200,

    max_depth=8,

    min_samples_leaf=10,

    class_weight="balanced",

    random_state=42,

    n_jobs=-1
)


# ============================================================
# 7. TRAIN MODEL
# ============================================================

print("\nTraining Random Forest model...")

model.fit(
    X_train,
    y_train
)

print("Model training completed.")


# ============================================================
# 8. EVALUATE MODEL
# ============================================================

y_pred = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\n" + "=" * 70)
print("MODEL EVALUATION")
print("=" * 70)

print(
    "Test Accuracy:",
    round(accuracy * 100, 2),
    "%"
)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "Not Placed",
            "Placed"
        ]
    )
)


# ============================================================
# 9. SAVE FINAL MODEL
# ============================================================

MODEL_FILE = "final_random_forest_model.pkl"

joblib.dump(
    model,
    MODEL_FILE
)

print("\nFinal model saved as:")

print(MODEL_FILE)


# ============================================================
# 10. USER INPUT
# ============================================================

print("\n" + "=" * 70)
print("ENTER STUDENT DETAILS")
print("=" * 70)

print("\nEnter the student's information below.")


iq = float(
    input("IQ (40-160): ")
)

prev_sem_result = float(
    input("Previous Semester Result (5-10): ")
)

cgpa = float(
    input("CGPA (5-10): ")
)

academic_performance = float(
    input("Academic Performance (1-10): ")
)

internship = input(
    "Internship Experience (Yes/No): "
).strip().lower()


extra_curricular = float(
    input("Extra-Curricular Score (0-10): ")
)

communication = float(
    input("Communication Skills (1-10): ")
)

projects = int(
    input("Projects Completed (0-5): ")
)


# ============================================================
# 11. CONVERT USER INPUT
# ============================================================

if internship == "yes":
    internship_value = 1
elif internship == "no":
    internship_value = 0
else:
    print("\nInvalid internship value.")
    print("Please enter Yes or No.")
    raise SystemExit


# ============================================================
# 12. CREATE INPUT DATAFRAME
# ============================================================

student_data = pd.DataFrame({

    "IQ": [iq],

    "Prev_Sem_Result": [
        prev_sem_result
    ],

    "CGPA": [cgpa],

    "Academic_Performance": [
        academic_performance
    ],

    "Internship_Experience": [
        internship_value
    ],

    "Extra_Curricular_Score": [
        extra_curricular
    ],

    "Communication_Skills": [
        communication
    ],

    "Projects_Completed": [
        projects
    ]
})


# ============================================================
# 13. MAKE PREDICTION
# ============================================================

prediction = model.predict(
    student_data
)[0]


probabilities = model.predict_proba(
    student_data
)[0]


not_placed_probability = probabilities[0]

placed_probability = probabilities[1]


# ============================================================
# 14. DISPLAY RESULT
# ============================================================

print("\n" + "=" * 70)
print("PLACEMENT PREDICTION RESULT")
print("=" * 70)


if prediction == 1:

    print("\nPrediction: PLACED")

else:

    print("\nPrediction: NOT PLACED")


print(
    "\nProbability of Placement:",
    round(
        placed_probability * 100,
        2
    ),
    "%"
)


print(
    "Probability of Not Placed:",
    round(
        not_placed_probability * 100,
        2
    ),
    "%"
)


print("\n" + "=" * 70)
print("PREDICTION COMPLETED")
print("=" * 70)