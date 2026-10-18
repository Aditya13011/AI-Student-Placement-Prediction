
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

st.set_page_config(
    page_title="Model Analytics",
    page_icon="📈",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "college_student_placement_dataset.csv"

FEATURES = [
    "IQ",
    "Prev_Sem_Result",
    "CGPA",
    "Academic_Performance",
    "Internship_Experience",
    "Extra_Curricular_Score",
    "Communication_Skills",
    "Projects_Completed"
]
TARGET = "Placement"

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}
.stApp {
    background: linear-gradient(135deg, #080d1b, #10172b);
    color: #eef2ff;
}
.block-container {
    max-width: 1400px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}
[data-testid="stSidebar"] {
    background: #0b1020;
    border-right: 1px solid #293653;
}
h1, h2, h3 {
    color: #f1f5ff !important;
}
[data-testid="stMetric"] {
    background: #151e33;
    border: 1px solid #293653;
    border-radius: 16px;
    padding: 18px;
}
[data-testid="stMetricValue"] {
    color: #8b9cff;
}
.analytics-hero {
    background: linear-gradient(120deg, #182443, #25204a);
    border: 1px solid #35436b;
    padding: 28px 32px;
    border-radius: 22px;
    margin-bottom: 24px;
}
.analytics-hero h1 {
    font-size: 32px;
    font-weight: 800;
    margin: 0;
}
.analytics-hero p {
    color: #b8c5e2;
    margin-bottom: 0;
}
.small-note {
    color: #aab8d3;
    font-size: 13px;
}
div[data-testid="stDataFrame"] {
    border: 1px solid #293653;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="analytics-hero">
    <h1>📈 Model Analytics</h1>
    <p>Evaluate classification performance and understand which features influence the model.</p>
</div>
""", unsafe_allow_html=True)

st.warning(
    "The metrics below are calculated from a newly trained Random Forest model "
    "using a hold-out test split. They do not automatically describe the exact "
    "performance of final_random_forest_model.pkl used on the prediction pages."
)

if not DATA_PATH.exists():
    st.error(f"Dataset not found: {DATA_PATH}")
    st.stop()


@st.cache_data
def load_data(path):
    return pd.read_csv(path)


try:
    df = load_data(str(DATA_PATH))
except Exception as exc:
    st.error(f"Could not load dataset: {exc}")
    st.stop()

missing = [col for col in FEATURES + [TARGET] if col not in df.columns]
if missing:
    st.error("Missing required columns: " + ", ".join(missing))
    st.stop()

df = df[FEATURES + [TARGET]].copy()

# Normalize common Yes/No and boolean encodings.
internship_map = {
    "yes": 1, "no": 0, "true": 1, "false": 0,
    "1": 1, "0": 0, "1.0": 1, "0.0": 0
}
df["Internship_Experience"] = (
    df["Internship_Experience"]
    .astype(str).str.strip().str.lower()
    .map(internship_map)
)

for col in FEATURES:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Support common binary target representations.
target_text = df[TARGET].astype(str).str.strip().str.lower()
target_map = {
    "yes": 1, "placed": 1, "true": 1, "1": 1, "1.0": 1,
    "no": 0, "not placed": 0, "false": 0, "0": 0, "0.0": 0
}
df[TARGET] = target_text.map(target_map)

before = len(df)
df = df.dropna(subset=FEATURES + [TARGET]).copy()
df[TARGET] = df[TARGET].astype(int)
dropped = before - len(df)

if dropped:
    st.info(f"{dropped} row(s) were excluded because required values were missing or invalid.")

if len(df) < 20:
    st.error("At least 20 valid rows are required for this evaluation.")
    st.stop()

if df[TARGET].nunique() < 2:
    st.error("The target column must contain both placement classes.")
    st.stop()

st.sidebar.header("Evaluation Settings")
test_size_percent = st.sidebar.slider(
    "Test set (%)",
    min_value=15,
    max_value=40,
    value=20,
    step=5
)
n_estimators = st.sidebar.slider(
    "Number of trees",
    min_value=50,
    max_value=300,
    value=100,
    step=50
)
random_state = st.sidebar.number_input(
    "Random seed",
    min_value=0,
    max_value=10000,
    value=42,
    step=1
)

X = df[FEATURES]
y = df[TARGET]

class_counts = y.value_counts()
can_stratify = class_counts.min() >= 2

try:
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size_percent / 100,
        random_state=int(random_state),
        stratify=y if can_stratify else None
    )

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=int(random_state),
        class_weight="balanced",
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

except Exception as exc:
    st.error(f"Model evaluation failed: {exc}")
    st.stop()

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

st.subheader("Test Set Performance")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Accuracy", f"{accuracy * 100:.2f}%")
m2.metric("Precision", f"{precision * 100:.2f}%")
m3.metric("Recall", f"{recall * 100:.2f}%")
m4.metric("F1 Score", f"{f1 * 100:.2f}%")

st.caption(
    f"Training rows: {len(X_train):,} · Test rows: {len(X_test):,} · "
    f"Random seed: {int(random_state)}"
)

st.divider()

left, right = st.columns(2)

with left:
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=["Predicted Not Placed", "Predicted Placed"],
        y=["Actual Not Placed", "Actual Placed"],
        text=cm,
        texttemplate="%{text}",
        colorscale=[
            [0, "#151e33"],
            [0.5, "#5367c5"],
            [1, "#8b9cff"]
        ],
        showscale=False
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e7edff",
        xaxis_title="Predicted Class",
        yaxis_title="Actual Class",
        margin=dict(t=25, b=20, l=10, r=10)
    )
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("Feature Importance")
    importance = pd.DataFrame({
        "Feature": FEATURES,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=True)

    fig = px.bar(
        importance,
        x="Importance",
        y="Feature",
        orientation="h",
        text=importance["Importance"].map(lambda x: f"{x:.3f}")
    )
    fig.update_traces(
        marker_color="#7c8cff",
        textposition="outside",
        cliponaxis=False
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e7edff",
        xaxis_title="Relative Importance",
        yaxis_title="",
        margin=dict(t=20, b=20, l=10, r=40)
    )
    fig.update_xaxes(gridcolor="#283550")
    st.plotly_chart(fig, use_container_width=True)

st.divider()
st.subheader("Classification Report")

report = classification_report(
    y_test,
    y_pred,
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

report_csv = report_df.to_csv().encode("utf-8")
st.download_button(
    "⬇️ Download Classification Report",
    data=report_csv,
    file_name="model_classification_report.csv",
    mime="text/csv"
)

st.markdown("---")
st.markdown(
    '<p class="small-note">Model Analytics · Test results depend on the dataset, split, preprocessing, and model configuration. Feature importance indicates model reliance, not causation.</p>',
    unsafe_allow_html=True
)
