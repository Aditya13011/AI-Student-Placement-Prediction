import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# 1. LOAD DATASET
# ============================================================

df = pd.read_csv("college_student_placement_dataset.csv")

print("=" * 70)
print("AI-BASED STUDENT PLACEMENT PREDICTION SYSTEM")
print("MODEL TRAINING")
print("=" * 70)

print("\nOriginal dataset shape:")
print(df.shape)


# ============================================================
# 2. REMOVE IDENTIFIER
# ============================================================

# College_ID is only an identifier.
# It should not be used for prediction.

df = df.drop(columns=["College_ID"])


# ============================================================
# 3. CONVERT CATEGORICAL VALUES
# ============================================================

# Internship Experience:
# Yes = 1
# No  = 0

df["Internship_Experience"] = df["Internship_Experience"].map({
    "Yes": 1,
    "No": 0
})


# Placement:
# Yes = 1
# No  = 0

df["Placement"] = df["Placement"].map({
    "Yes": 1,
    "No": 0
})


# ============================================================
# 4. CHECK MISSING VALUES
# ============================================================

print("\nMissing values before cleaning:")

print(df.isnull().sum())


# ============================================================
# 5. HANDLE MISSING NUMERIC VALUES
# ============================================================

numeric_columns = df.select_dtypes(
    include=["int64", "float64"]
).columns

for column in numeric_columns:

    df[column] = df[column].fillna(
        df[column].median()
    )


print("\nMissing values after cleaning:")

print(df.isnull().sum())


# ============================================================
# 6. SEPARATE FEATURES AND TARGET
# ============================================================

X = df.drop(columns=["Placement"])

y = df["Placement"]


print("\nFeatures used for training:")

print(X.columns.tolist())

print("\nTarget variable:")

print("Placement")


# ============================================================
# 7. TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42,

    # Keeps the Yes/No ratio similar
    # in training and testing data.
    stratify=y
)


print("\nTraining data shape:")

print(X_train.shape)


print("\nTesting data shape:")

print(X_test.shape)


print("\nTraining target distribution:")

print(y_train.value_counts())


print("\nTesting target distribution:")

print(y_test.value_counts())


# ============================================================
# 8. DEFINE MACHINE LEARNING MODELS
# ============================================================

models = {

    # --------------------------------------------------------
    # Logistic Regression
    # --------------------------------------------------------

    "Logistic Regression": Pipeline([

        ("scaler", StandardScaler()),

        ("model", LogisticRegression(

            max_iter=1000,

            # Important because our dataset is imbalanced
            class_weight="balanced",

            random_state=42
        ))
    ]),


    # --------------------------------------------------------
    # Decision Tree
    # --------------------------------------------------------

    "Decision Tree": DecisionTreeClassifier(

        max_depth=6,

        min_samples_leaf=15,

        class_weight="balanced",

        random_state=42
    ),


    # --------------------------------------------------------
    # Random Forest
    # --------------------------------------------------------

    "Random Forest": RandomForestClassifier(

        n_estimators=200,

        max_depth=8,

        min_samples_leaf=10,

        class_weight="balanced",

        random_state=42,

        n_jobs=-1
    ),


    # --------------------------------------------------------
    # K-Nearest Neighbors
    # --------------------------------------------------------

    "KNN": Pipeline([

        ("scaler", StandardScaler()),

        ("model", KNeighborsClassifier(

            n_neighbors=15
        ))
    ]),


    # --------------------------------------------------------
    # Support Vector Machine
    # --------------------------------------------------------

    "SVM": Pipeline([

        ("scaler", StandardScaler()),

        ("model", SVC(

            probability=True,

            class_weight="balanced",

            random_state=42
        ))
    ])
}


# ============================================================
# 9. TRAIN AND EVALUATE MODELS
# ============================================================

results = []


for name, model in models.items():

    print("\n")
    print("=" * 70)

    print("TRAINING:", name)

    print("=" * 70)


    # --------------------------------------------------------
    # Train model
    # --------------------------------------------------------

    model.fit(

        X_train,

        y_train
    )


    # --------------------------------------------------------
    # Make predictions
    # --------------------------------------------------------

    y_pred = model.predict(X_test)


    # --------------------------------------------------------
    # Get prediction probabilities
    # --------------------------------------------------------

    y_probability = model.predict_proba(X_test)[:, 1]


    # --------------------------------------------------------
    # Calculate metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(

        y_test,

        y_pred
    )


    precision = precision_score(

        y_test,

        y_pred,

        zero_division=0
    )


    recall = recall_score(

        y_test,

        y_pred,

        zero_division=0
    )


    f1 = f1_score(

        y_test,

        y_pred,

        zero_division=0
    )


    roc_auc = roc_auc_score(

        y_test,

        y_probability
    )


    # --------------------------------------------------------
    # Store results
    # --------------------------------------------------------

    results.append({

        "Model": name,

        "Accuracy": accuracy,

        "Precision": precision,

        "Recall": recall,

        "F1_Score": f1,

        "ROC_AUC": roc_auc
    })


    # --------------------------------------------------------
    # Display results
    # --------------------------------------------------------

    print("\nAccuracy:")

    print(round(accuracy, 4))


    print("\nPrecision:")

    print(round(precision, 4))


    print("\nRecall:")

    print(round(recall, 4))


    print("\nF1 Score:")

    print(round(f1, 4))


    print("\nROC-AUC:")

    print(round(roc_auc, 4))


    # --------------------------------------------------------
    # Confusion Matrix
    # --------------------------------------------------------

    print("\nConfusion Matrix:")

    print(
        confusion_matrix(
            y_test,
            y_pred
        )
    )


    # --------------------------------------------------------
    # Classification Report
    # --------------------------------------------------------

    print("\nClassification Report:")

    print(
        classification_report(

            y_test,

            y_pred,

            target_names=[
                "Not Placed",
                "Placed"
            ],

            zero_division=0
        )
    )


# ============================================================
# 10. MODEL COMPARISON
# ============================================================

comparison_df = pd.DataFrame(results)


print("\n\n")

print("=" * 70)

print("MODEL COMPARISON")

print("=" * 70)


print(
    comparison_df.to_string(

        index=False,

        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# 11. SELECT BEST MODEL
# ============================================================

# The dataset is imbalanced.
#
# Therefore, we use F1-score as the main metric
# instead of selecting a model only by accuracy.

best_index = comparison_df[
    "F1_Score"
].idxmax()


best_model_name = comparison_df.loc[
    best_index,
    "Model"
]


print("\n")

print("=" * 70)

print("BEST MODEL")

print("=" * 70)


print(
    "Selected model:",
    best_model_name
)


print(
    "F1 Score:",
    round(
        comparison_df.loc[
            best_index,
            "F1_Score"
        ],
        4
    )
)


print(
    "ROC-AUC:",
    round(
        comparison_df.loc[
            best_index,
            "ROC_AUC"
        ],
        4
    )
)


# ============================================================
# 12. SAVE MODEL COMPARISON
# ============================================================

comparison_df.to_csv(

    "model_comparison.csv",

    index=False
)


print("\nModel comparison saved as:")

print("model_comparison.csv")


# ============================================================
# 13. SAVE THE BEST MODEL
# ============================================================

best_model = models[best_model_name]


joblib.dump(

    best_model,

    "best_placement_model.pkl"
)


print("\nBest model saved as:")

print("best_placement_model.pkl")


# ============================================================
# 14. FINISH
# ============================================================

print("\n")

print("=" * 70)

print("MODEL TRAINING COMPLETED SUCCESSFULLY")

print("=" * 70)