import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


# ============================================================
# 1. LOAD DATASET
# ============================================================

df = pd.read_csv(
    "college_student_placement_dataset.csv"
)


# ============================================================
# 2. PREPARE DATA
# ============================================================

df = df.drop(
    columns=["College_ID"]
)

df["Internship_Experience"] = df[
    "Internship_Experience"
].map({
    "Yes": 1,
    "No": 0
})

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
# 5. DEFINE MODELS
# ============================================================

models = {

    "Logistic Regression": Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
        (
            "model",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=42
            )
        )
    ]),

    "Decision Tree": DecisionTreeClassifier(
        max_depth=6,
        min_samples_leaf=15,
        class_weight="balanced",
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=10,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),

    "KNN": Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
        (
            "model",
            KNeighborsClassifier(
                n_neighbors=15
            )
        )
    ]),

    "SVM": Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
        (
            "model",
            SVC(
                probability=True,
                class_weight="balanced",
                random_state=42
            )
        )
    ])
}


# ============================================================
# 6. STRATIFIED 5-FOLD CROSS VALIDATION
# ============================================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# ============================================================
# 7. EVALUATE MODELS
# ============================================================

results = []


for name, model in models.items():

    print("\n" + "=" * 70)

    print(
        "CROSS-VALIDATING:",
        name
    )

    print("=" * 70)


    scores = cross_validate(

        model,

        X,

        y,

        cv=cv,

        scoring=[
            "accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc"
        ],

        n_jobs=-1
    )


    accuracy = scores[
        "test_accuracy"
    ]

    precision = scores[
        "test_precision"
    ]

    recall = scores[
        "test_recall"
    ]

    f1 = scores[
        "test_f1"
    ]

    roc_auc = scores[
        "test_roc_auc"
    ]


    print(
        "Accuracy:",
        round(accuracy.mean(), 4),
        "+/-",
        round(accuracy.std(), 4)
    )

    print(
        "Precision:",
        round(precision.mean(), 4),
        "+/-",
        round(precision.std(), 4)
    )

    print(
        "Recall:",
        round(recall.mean(), 4),
        "+/-",
        round(recall.std(), 4)
    )

    print(
        "F1:",
        round(f1.mean(), 4),
        "+/-",
        round(f1.std(), 4)
    )

    print(
        "ROC-AUC:",
        round(roc_auc.mean(), 4),
        "+/-",
        round(roc_auc.std(), 4)
    )


    results.append({

        "Model": name,

        "Accuracy": accuracy.mean(),

        "Precision": precision.mean(),

        "Recall": recall.mean(),

        "F1_Score": f1.mean(),

        "ROC_AUC": roc_auc.mean()
    })


# ============================================================
# 8. COMPARISON
# ============================================================

results_df = pd.DataFrame(
    results
)


print("\n\n")

print("=" * 70)

print("5-FOLD CROSS-VALIDATION RESULTS")

print("=" * 70)


print(
    results_df.to_string(
        index=False,
        float_format=lambda x:
        f"{x:.4f}"
    )
)


# ============================================================
# 9. SAVE RESULTS
# ============================================================

results_df.to_csv(
    "cross_validation_results.csv",
    index=False
)


print("\nResults saved as:")

print("cross_validation_results.csv")


print("\n" + "=" * 70)

print("CROSS-VALIDATION COMPLETED")

print("=" * 70)