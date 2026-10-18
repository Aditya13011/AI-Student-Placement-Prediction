
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(
    page_title="Batch Prediction",
    page_icon="📂",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "final_random_forest_model.pkl"

FEATURES = [
    "IQ",
    "Prev_Sem_Result",
    "CGPA",
    "Academic_Performance",
    "Internship_Experience",
    "Extra_Curricular_Score",
    "Communication_Skills",
    "Projects_Completed",
]

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
    max-width: 1350px;
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
.batch-hero {
    background: linear-gradient(120deg, #182443, #25204a);
    border: 1px solid #35436b;
    padding: 28px 32px;
    border-radius: 22px;
    margin-bottom: 24px;
}
.batch-hero h1 {
    font-size: 32px;
    font-weight: 800;
    margin: 0;
}
.batch-hero p {
    color: #b8c5e2;
    margin-bottom: 0;
}
.upload-card {
    background: #111a2d;
    border: 1px solid #293653;
    border-radius: 16px;
    padding: 20px;
    margin-bottom: 18px;
}
div.stButton > button, div.stDownloadButton > button {
    background: linear-gradient(90deg, #697bff, #8b5cf6);
    color: white;
    border: 0;
    border-radius: 10px;
    font-weight: 700;
}
.small-note {
    color: #aab8d3;
    font-size: 13px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="batch-hero">
    <h1>📂 Batch Student Prediction</h1>
    <p>Upload student records and generate placement predictions in one workflow.</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="upload-card">
<strong>Required CSV columns</strong>
<p class="small-note">
IQ, Prev_Sem_Result, CGPA, Academic_Performance,
Internship_Experience, Extra_Curricular_Score,
Communication_Skills, Projects_Completed
</p>
</div>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model(path):
    return joblib.load(path)


if not MODEL_PATH.exists():
    st.error(f"Model file not found: {MODEL_PATH}")
    st.info("Place final_random_forest_model.pkl in your project root folder.")
    st.stop()

try:
    model = load_model(str(MODEL_PATH))
except Exception as exc:
    st.error(f"Could not load the saved model: {exc}")
    st.stop()

uploaded_file = st.file_uploader(
    "Upload student CSV file",
    type=["csv"],
    help="Upload a CSV with all eight required feature columns."
)

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)
    except Exception as exc:
        st.error(f"Could not read this CSV file: {exc}")
        st.stop()

    st.subheader("Uploaded Data Preview")
    st.caption(f"{len(data):,} student records detected.")
    st.dataframe(data.head(10), use_container_width=True, hide_index=True)

    missing = [col for col in FEATURES if col not in data.columns]

    if missing:
        st.error(
            "Missing required columns: " + ", ".join(missing)
        )
        st.stop()

    if data.empty:
        st.warning("The uploaded CSV contains no student records.")
        st.stop()

    if st.button("🚀 Generate Predictions", use_container_width=True):
        try:
            input_data = data[FEATURES].copy()

            # Convert Yes/No internship values into 1/0.
            internship_values = (
                input_data["Internship_Experience"]
                .astype(str).str.strip().str.lower()
            )
            internship_map = {
                "yes": 1, "no": 0,
                "true": 1, "false": 0,
                "1": 1, "0": 0,
                "1.0": 1, "0.0": 0
            }
            input_data["Internship_Experience"] = (
                internship_values.map(internship_map)
                .fillna(input_data["Internship_Experience"])
            )

            for column in FEATURES:
                input_data[column] = pd.to_numeric(
                    input_data[column], errors="coerce"
                )

            invalid_rows = input_data[FEATURES].isna().any(axis=1)

            if invalid_rows.any():
                st.error(
                    f"{int(invalid_rows.sum())} row(s) contain missing "
                    "or invalid feature values. Correct the CSV and upload it again."
                )
                st.stop()

            if hasattr(model, "n_features_in_"):
                if model.n_features_in_ != len(FEATURES):
                    st.error(
                        "The saved model does not expect eight input features. "
                        "Check the model training pipeline before using this page."
                    )
                    st.stop()

            predictions = model.predict(input_data[FEATURES])

            results = data.copy()
            results["Predicted_Placement"] = [
                "Placed" if str(value).strip().lower() in
                ["1", "1.0", "true", "yes", "placed"]
                else "Not Placed"
                for value in predictions
            ]

            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(input_data[FEATURES])

                classes = list(getattr(model, "classes_", []))
                positive_index = None

                for index, label in enumerate(classes):
                    if str(label).strip().lower() in [
                        "1", "1.0", "true", "yes", "placed"
                    ]:
                        positive_index = index
                        break

                if positive_index is not None:
                    results["Placement_Probability_%"] = (
                        probabilities[:, positive_index] * 100
                    ).round(2)

            st.session_state["batch_results"] = results

        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
            st.info(
                "Check that your CSV uses the same feature names, order, "
                "data types, and preprocessing as the saved model."
            )

if "batch_results" in st.session_state:
    results = st.session_state["batch_results"]

    st.divider()
    st.subheader("Prediction Results")

    total = len(results)
    placed = int((results["Predicted_Placement"] == "Placed").sum())
    not_placed = total - placed
    rate = placed / total * 100 if total else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Students", f"{total:,}")
    c2.metric("Predicted Placed", f"{placed:,}")
    c3.metric("Predicted Not Placed", f"{not_placed:,}")
    c4.metric("Predicted Placement Rate", f"{rate:.1f}%")

    st.dataframe(
        results,
        use_container_width=True,
        hide_index=True
    )

    csv_bytes = results.to_csv(index=False).encode("utf-8")

    st.download_button(
        "⬇️ Download Prediction Results",
        data=csv_bytes,
        file_name="student_placement_predictions.csv",
        mime="text/csv",
        use_container_width=True
    )

    st.caption(
        "Predictions are estimates from the saved model, not guarantees of actual placement."
    )

st.markdown("---")
st.markdown(
    '<p class="small-note">Batch Prediction · Review the uploaded data and model compatibility before interpreting results.</p>',
    unsafe_allow_html=True
)
