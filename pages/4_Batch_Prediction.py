
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(
    page_title="Batch Placement Prediction",
    page_icon="📂",
    layout="wide"
)

st.title("📂 Batch Placement Prediction")
st.write(
    "Upload a CSV containing multiple student profiles "
    "to generate placement predictions."
)

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "final_random_forest_model.pkl"

REQUIRED_COLUMNS = [
    "IQ",
    "Prev_Sem_Result",
    "CGPA",
    "Academic_Performance",
    "Internship_Experience",
    "Extra_Curricular_Score",
    "Communication_Skills",
    "Projects_Completed"
]

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

def normalize_internship(series):
    mapping = {
        "yes": 1,
        "no": 0,
        "1": 1,
        "0": 0,
        "true": 1,
        "false": 0
    }

    normalized = series.astype(str).str.strip().str.lower()
    return normalized.map(mapping)

if not MODEL_PATH.exists():
    st.error(
        "Model file not found: final_random_forest_model.pkl. "
        "Place the model file in the project root directory."
    )
    st.stop()

model = load_model()

st.subheader("1. Upload Student Data")

st.write("Required columns:")
st.code(", ".join(REQUIRED_COLUMNS))

st.download_button(
    "⬇️ Download Sample CSV Template",
    data=pd.DataFrame({
        "IQ": [110, 95],
        "Prev_Sem_Result": [75, 65],
        "CGPA": [8.2, 6.8],
        "Academic_Performance": [8, 6],
        "Internship_Experience": ["Yes", "No"],
        "Extra_Curricular_Score": [7, 5],
        "Communication_Skills": [8, 6],
        "Projects_Completed": [3, 1]
    }).to_csv(index=False).encode("utf-8"),
    file_name="student_batch_template.csv",
    mime="text/csv"
)

uploaded_file = st.file_uploader(
    "Choose a student CSV file",
    type=["csv"]
)

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)

        st.subheader("2. Preview Uploaded Data")
        st.dataframe(data.head(10), use_container_width=True)

        missing_columns = [
            column for column in REQUIRED_COLUMNS
            if column not in data.columns
        ]

        if missing_columns:
            st.error(
                "Missing required columns: "
                + ", ".join(missing_columns)
            )
            st.stop()

        if data.empty:
            st.warning("The uploaded CSV contains no student records.")
            st.stop()

        if st.button(
            "🚀 Predict All Students",
            type="primary",
            use_container_width=True
        ):
            input_data = data[REQUIRED_COLUMNS].copy()

            internship_values = normalize_internship(
                input_data["Internship_Experience"]
            )

            if internship_values.isna().any():
                st.error(
                    "Internship_Experience must contain Yes/No, "
                    "True/False, or 1/0 values."
                )
                st.stop()

            input_data["Internship_Experience"] = (
                internship_values.astype(int)
            )

            for column in REQUIRED_COLUMNS:
                if column != "Internship_Experience":
                    input_data[column] = pd.to_numeric(
                        input_data[column],
                        errors="coerce"
                    )

            if input_data.isna().any().any():
                invalid_columns = input_data.columns[
                    input_data.isna().any()
                ].tolist()

                st.error(
                    "Missing or non-numeric values found in: "
                    + ", ".join(invalid_columns)
                    + ". Correct the CSV and upload it again."
                )
                st.stop()

            if not hasattr(model, "n_features_in_"):
                st.error(
                    "The saved model does not expose its expected "
                    "feature count. Verify the model before prediction."
                )
                st.stop()

            if model.n_features_in_ != len(REQUIRED_COLUMNS):
                st.error(
                    f"The model expects {model.n_features_in_} features, "
                    f"but this page supplies {len(REQUIRED_COLUMNS)}. "
                    "The model's original training feature order must "
                    "be verified before using batch prediction."
                )
                st.stop()

            try:
                predictions = model.predict(input_data)

                results = data.copy()
                results["Predicted_Placement"] = predictions

                if hasattr(model, "predict_proba"):
                    probabilities = model.predict_proba(input_data)
                    classes = list(model.classes_)

                    yes_class = next(
                        (
                            cls for cls in classes
                            if str(cls).strip().lower()
                            in ["yes", "1", "true"]
                        ),
                        None
                    )

                    if yes_class is not None:
                        yes_index = classes.index(yes_class)
                        results["Placement_Probability_%"] = (
                            probabilities[:, yes_index] * 100
                        ).round(2)

                st.subheader("3. Prediction Results")

                st.dataframe(
                    results,
                    use_container_width=True
                )

                st.download_button(
                    "⬇️ Download Prediction Results",
                    data=results.to_csv(index=False).encode("utf-8"),
                    file_name="placement_predictions.csv",
                    mime="text/csv"
                )

                st.success(
                    f"Processed {len(results)} student records."
                )

                st.caption(
                    "Predictions and probabilities are model outputs, "
                    "not guarantees of actual placement."
                )

            except Exception as error:
                st.error(
                    "Prediction failed. The uploaded feature format "
                    "may not match the model's training format."
                )
                st.code(str(error))

    except Exception as error:
        st.error(f"Unable to read the CSV file: {error}")

st.divider()
st.caption("Student Placement Intelligence System | Batch Prediction")
