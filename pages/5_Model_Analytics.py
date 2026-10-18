
import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from sklearn.preprocessing import LabelEncoder

st.set_page_config(
    page_title="Model Analytics",
    page_icon="🤖",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "college_student_placement_dataset.csv"
MODEL_PATH = BASE_DIR / "final_random_forest_model.pkl"

st.title("🤖 Machine Learning Model Analytics")
st.caption("Evaluate model performance and understand important features.")
st.divider()


@st.cache_data
def load_dataset(path):
    return pd.read_csv(path)


try:
    df = load_dataset(DATA_PATH)
except Exception as error:
    st.error(f"Could not load dataset: {error}")
    st.stop()

if "Placement" not in df.columns:
    st.error("The dataset must contain a Placement column.")
    st.stop()

# Keep only valid target records.
df["Placement"] = (
    df["Placement"].astype(str).str.strip().str.title()
)
df = df[df["Placement"].isin(["Yes", "No"])].copy()

if df.empty:
    st.error("No valid placement labels were found.")
    st.stop()

# Remove identifiers and target from the predictors.
# College_ID is an identifier, not a student performance feature.
X = df.drop(
    columns=["Placement", "College_ID"],
    errors="ignore"
).copy()

y = df["Placement"].map({"No": 0, "Yes": 1})

# Convert categorical columns into numeric columns.
for column in X.select_dtypes(include=["object", "string"]).columns:
    X[column] = LabelEncoder().fit_transform(
        X[column].fillna("Missing").astype(str)
    )

X = X.apply(pd.to_numeric, errors="coerce")
X = X.replace([float("inf"), float("-inf")], float("nan"))

# Remove columns that are entirely missing.
X = X.dropna(axis=1, how="all")

# Median-impute remaining missing feature values.
X = X.fillna(X.median()).fillna(0)

if X.empty:
    st.error("No usable predictor columns remain.")
    st.stop()

# Create a reproducible holdout split.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

st.subheader("Evaluation setup")
st.write(
    f"Dataset: **{len(df):,} rows** | "
    f"Training rows: **{len(X_train):,}** | "
    f"Test rows: **{len(X_test):,}**"
)

st.warning(
    "This page trains a fresh Random Forest on a holdout split. "
    "It does not assume the saved model was trained using the same "
    "preprocessing. Treat results as diagnostic until the training "
    "and inference pipeline is standardized."
)

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)

try:
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test, predictions, zero_division=0
    )
    recall = recall_score(
        y_test, predictions, zero_division=0
    )
    f1 = f1_score(
        y_test, predictions, zero_division=0
    )

except Exception as error:
    st.error(f"Model evaluation failed: {error}")
    st.stop()

st.subheader("Performance metrics")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Accuracy", f"{accuracy * 100:.2f}%")
c2.metric("Precision", f"{precision * 100:.2f}%")
c3.metric("Recall", f"{recall * 100:.2f}%")
c4.metric("F1-score", f"{f1 * 100:.2f}%")

st.caption(
    "Accuracy is the overall fraction of correct predictions. "
    "Precision and recall focus on the positive (Placed) class. "
    "F1-score balances precision and recall."
)

left, right = st.columns(2)

with left:
    st.subheader("Confusion matrix")

    cm = confusion_matrix(
        y_test, predictions, labels=[0, 1]
    )

    cm_df = pd.DataFrame(
        cm,
        index=["Actual Not Placed", "Actual Placed"],
        columns=["Predicted Not Placed", "Predicted Placed"]
    )

    fig = px.imshow(
        cm_df,
        text_auto=True,
        aspect="auto",
        title="Actual vs Predicted"
    )

    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("Classification report")

    report = classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["Not Placed", "Placed"],
        output_dict=True,
        zero_division=0
    )

    report_df = pd.DataFrame(report).transpose()
    st.dataframe(
        report_df.round(3),
        use_container_width=True
    )

st.subheader("Feature importance")

importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
}).sort_values("Importance", ascending=False)

fig = px.bar(
    importance_df,
    x="Importance",
    y="Feature",
    orientation="h",
    title="Random Forest Feature Importance"
)
fig.update_layout(yaxis={"categoryorder": "total ascending"})
st.plotly_chart(fig, use_container_width=True)

st.dataframe(
    importance_df.round(4),
    use_container_width=True,
    hide_index=True
)

with st.expander("View test predictions"):
    results = pd.DataFrame({
        "Actual": y_test.map({0: "No", 1: "Yes"}),
        "Predicted": pd.Series(
            predictions, index=y_test.index
        ).map({0: "No", 1: "Yes"})
    })

    st.dataframe(results, use_container_width=True)

    st.download_button(
        "Download test predictions",
        data=results.to_csv(index=True).encode("utf-8"),
        file_name="model_test_predictions.csv",
        mime="text/csv"
    )

st.divider()
st.caption(
    "Student Placement Intelligence System | Model Analytics"
)
