
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# =========================================================
# PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="Placement Intelligence | AI Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "final_random_forest_model.pkl"

# =========================================================
# PREMIUM DARK UI
# =========================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
    --bg: #0a0f1e;
    --surface: #121a2c;
    --surface-2: #18233a;
    --border: #293650;
    --text: #eef2ff;
    --muted: #98a8c4;
    --accent: #818cf8;
    --accent-2: #6366f1;
    --green: #34d399;
    --red: #fb7185;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(ellipse at top right, #1a2450 0%, transparent 38%),
        linear-gradient(145deg, #090e1b, #0e1628 55%, #0a1020);
    color: var(--text);
}

.block-container {
    max-width: 1450px;
    padding-top: 2rem;
    padding-bottom: 2rem;
}

h1, h2, h3, h4 {
    color: var(--text) !important;
    letter-spacing: -0.5px;
}

p, label, .stMarkdown {
    color: #c5cfe2;
}

[data-testid="stSidebar"] {
    background: #0b1120;
    border-right: 1px solid var(--border);
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] p {
    color: var(--text) !important;
}

[data-testid="stSidebar"] .stCaption {
    color: var(--muted) !important;
}

.hero {
    padding: 30px 32px;
    border: 1px solid #344267;
    border-radius: 22px;
    background: linear-gradient(120deg, #192747 0%, #171b3a 58%, #20234c 100%);
    margin-bottom: 25px;
    box-shadow: 0 16px 45px rgba(0, 0, 0, 0.20);
}

.eyebrow {
    color: #a5b4fc;
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 12px;
}

.hero h1 {
    font-size: clamp(27px, 4vw, 43px);
    font-weight: 800;
    margin: 0 0 12px 0;
    line-height: 1.2;
}

.hero p {
    color: #c1cbe2;
    font-size: 15px;
    line-height: 1.8;
    margin: 0;
    max-width: 780px;
}

.section-heading {
    font-size: 22px;
    font-weight: 750;
    margin: 20px 0 6px 0;
    color: #f1f5ff;
}

.section-subtitle {
    font-size: 13px;
    color: var(--muted);
    margin-bottom: 18px;
}

.panel {
    background: linear-gradient(145deg, #151f33, #111a2b);
    border: 1px solid var(--border);
    border-radius: 17px;
    padding: 21px;
    margin-bottom: 16px;
}

.panel-title {
    color: #e8edff;
    font-size: 16px;
    font-weight: 700;
    margin-bottom: 5px;
}

.panel-subtitle {
    color: var(--muted);
    font-size: 12px;
    margin-bottom: 15px;
}

[data-testid="stMetric"] {
    background: linear-gradient(145deg, #17243b, #121b2d);
    border: 1px solid #2c3a56;
    padding: 18px;
    border-radius: 15px;
}

[data-testid="stMetricLabel"] {
    color: #a9b8d2 !important;
}

[data-testid="stMetricValue"] {
    color: #a5b4fc !important;
    font-weight: 800;
}

.stTextInput input,
.stNumberInput input,
.stTextArea textarea {
    background: #0f1728 !important;
    color: #f1f5ff !important;
    border: 1px solid #34435f !important;
    border-radius: 10px !important;
}

.stSelectbox div[data-baseweb="select"] {
    background: #111a2c;
    border-radius: 10px;
}

.stButton > button,
.stDownloadButton > button {
    min-height: 44px;
    border-radius: 10px;
    border: 1px solid #6674ef;
    background: linear-gradient(110deg, #5865e9, #795cf5);
    color: white !important;
    font-weight: 700;
    transition: all 0.2s ease;
}

.stButton > button:hover,
.stDownloadButton > button:hover {
    border-color: #b4bdff;
    box-shadow: 0 5px 20px rgba(99, 102, 241, 0.24);
    transform: translateY(-1px);
}

div[data-testid="stProgressBar"] > div > div {
    background: linear-gradient(90deg, #6366f1, #a78bfa);
}

.result-box {
    border-radius: 18px;
    padding: 25px;
    margin: 15px 0;
    border: 1px solid;
    text-align: center;
}

.placed-box {
    background: linear-gradient(120deg, #102c2a, #12352e);
    border-color: #287d67;
}

.not-placed-box {
    background: linear-gradient(120deg, #321b2b, #2c1a2b);
    border-color: #8e435d;
}

.result-label {
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 2px;
    font-weight: 800;
    margin-bottom: 8px;
    color: #c3cee4;
}

.result-title {
    font-size: 32px;
    font-weight: 800;
    margin-bottom: 10px;
    color: #ffffff;
}

.result-description {
    font-size: 14px;
    color: #d0d9eb;
    line-height: 1.7;
}

.profile-item {
    background: #111b2e;
    border: 1px solid #293750;
    padding: 15px;
    border-radius: 12px;
    margin-bottom: 9px;
}

.profile-label {
    color: #96a8c7;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.7px;
    margin-bottom: 6px;
}

.profile-value {
    color: #f0f4ff;
    font-size: 17px;
    font-weight: 700;
}

.footer {
    text-align: center;
    color: #7585a2;
    border-top: 1px solid #253149;
    margin-top: 38px;
    padding: 22px 0 5px 0;
    font-size: 12px;
}

hr {
    border-color: #293650;
}

[data-testid="stAlert"] {
    border-radius: 12px;
}

@media (max-width: 768px) {
    .hero {
        padding: 22px;
    }
    .result-title {
        font-size: 25px;
    }
}
</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD EXISTING MODEL
# =========================================================
@st.cache_resource
def load_model(model_path):
    return joblib.load(model_path)


if not MODEL_PATH.exists():
    st.error("The saved model file was not found.")
    st.write(
        "Expected location: "
        f"`{MODEL_PATH}`"
    )
    st.info(
        "Place final_random_forest_model.pkl in the same "
        "folder as app.py."
    )
    st.stop()

try:
    model = load_model(str(MODEL_PATH))
except Exception as error:
    st.error("Unable to load the saved Random Forest model.")
    st.exception(error)
    st.stop()


# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.markdown("## 🎓 Placement AI")
    st.caption("STUDENT INTELLIGENCE PLATFORM")
    st.divider()

    st.markdown("### About the Project")
    st.write(
        "Estimate student placement outcomes using academic "
        "performance, communication, projects and internship "
        "experience."
    )

    st.divider()
    st.markdown("### Model")
    st.write("🌲 Random Forest Classifier")

    st.markdown("### Core Features")
    st.write("• Student prediction")
    st.write("• Placement probability")
    st.write("• Student profile summary")

    st.divider()
    st.caption("Machine Learning Mini Project")
    st.caption("Educational use only")


# =========================================================
# HERO HEADER
# =========================================================
st.markdown("""
<div class="hero">
    <div class="eyebrow">AI-powered student analytics</div>
    <h1>Student Placement Predictor</h1>
    <p>
        Understand a student's placement prediction through
        academic performance, practical experience and skill
        indicators. Generate a profile assessment in seconds.
    </p>
</div>
""", unsafe_allow_html=True)


# =========================================================
# INPUT SECTION
# =========================================================
st.markdown(
    '<div class="section-heading">Student Profile</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="section-subtitle">'
    'Enter the student details below to generate a model prediction.'
    '</div>',
    unsafe_allow_html=True
)

with st.container(border=True):
    st.markdown("#### 🧠 Academic & Aptitude Details")

    col1, col2, col3 = st.columns(3)

    with col1:
        iq = st.number_input(
            "IQ",
            min_value=40,
            max_value=160,
            value=120,
            step=1,
            help="Enter the IQ value used by your dataset."
        )

    with col2:
        prev_sem_result = st.number_input(
            "Previous Semester Result",
            min_value=5.0,
            max_value=10.0,
            value=8.0,
            step=0.1
        )

    with col3:
        cgpa = st.number_input(
            "CGPA",
            min_value=4.5,
            max_value=10.0,
            value=8.0,
            step=0.1
        )

    st.divider()
    st.markdown("#### 💼 Skills & Experience")

    col1, col2, col3 = st.columns(3)

    with col1:
        academic_performance = st.number_input(
            "Academic Performance",
            min_value=1.0,
            max_value=10.0,
            value=8.0,
            step=0.1
        )

    with col2:
        internship = st.selectbox(
            "Internship Experience",
            ["Yes", "No"]
        )

    with col3:
        extracurricular = st.number_input(
            "Extra-Curricular Score",
            min_value=1.0,
            max_value=10.0,
            value=7.0,
            step=0.1
        )

    col1, col2 = st.columns(2)

    with col1:
        communication = st.number_input(
            "Communication Skills",
            min_value=1.0,
            max_value=10.0,
            value=8.0,
            step=0.1
        )

    with col2:
        projects = st.number_input(
            "Projects Completed",
            min_value=0,
            max_value=5,
            value=3,
            step=1
        )

    st.markdown("")
    predict_button = st.button(
        "✨  Analyze Placement Profile",
        type="primary",
        use_container_width=True
    )


# =========================================================
# PREDICTION LOGIC
# =========================================================
if predict_button:
    internship_value = 1 if internship == "Yes" else 0

    # Keep the original feature names and order.
    input_data = pd.DataFrame([{
        "IQ": iq,
        "Prev_Sem_Result": prev_sem_result,
        "CGPA": cgpa,
        "Academic_Performance": academic_performance,
        "Internship_Experience": internship_value,
        "Extra_Curricular_Score": extracurricular,
        "Communication_Skills": communication,
        "Projects_Completed": projects
    }])

    try:
        prediction = model.predict(input_data)[0]

        classes = list(getattr(model, "classes_", []))
        probabilities = None

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_data)[0]

        def is_placed_class(value):
            return str(value).strip().lower() in {
                "1", "yes", "placed", "true"
            }

        if classes:
            placed_index = next(
                (i for i, value in enumerate(classes)
                 if is_placed_class(value)),
                None
            )
        else:
            placed_index = None

        placed_probability = None
        not_placed_probability = None

        if probabilities is not None and len(probabilities) == 2:
            if placed_index is not None:
                placed_probability = float(
                    probabilities[placed_index] * 100
                )
                other_index = 1 - placed_index
                not_placed_probability = float(
                    probabilities[other_index] * 100
                )
            elif set(map(str, classes)) == {"0", "1"}:
                placed_index = list(map(str, classes)).index("1")
                placed_probability = float(
                    probabilities[placed_index] * 100
                )
                not_placed_probability = 100.0 - placed_probability

        result_text = (
            "PLACED" if is_placed_class(prediction)
            else "NOT PLACED"
        )

        if result_text == "PLACED":
            result_class = "placed-box"
            result_icon = "✓"
            description = (
                "The model predicts a positive placement outcome "
                "for this student profile."
            )
        else:
            result_class = "not-placed-box"
            result_icon = "!"
            description = (
                "The model predicts a negative placement outcome "
                "for this student profile."
            )

        # -----------------------------------------------------
        # RESULT
        # -----------------------------------------------------
        st.markdown("---")
        st.markdown(
            '<div class="section-heading">Prediction Results</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            f"""
            <div class="result-box {result_class}">
                <div class="result-label">Model Assessment</div>
                <div class="result-title">{result_icon} {result_text}</div>
                <div class="result-description">{description}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        metric_col1, metric_col2 = st.columns(2)

        with metric_col1:
            if placed_probability is not None:
                st.metric(
                    "Placement Probability",
                    f"{placed_probability:.2f}%"
                )
            else:
                st.metric("Placement Probability", "Unavailable")

        with metric_col2:
            if not_placed_probability is not None:
                st.metric(
                    "Not Placed Probability",
                    f"{not_placed_probability:.2f}%"
                )
            else:
                st.metric("Not Placed Probability", "Unavailable")

        if placed_probability is not None:
            st.markdown("#### Placement Probability Indicator")
            st.progress(
                max(0.0, min(1.0, placed_probability / 100))
            )

        # -----------------------------------------------------
        # STUDENT PROFILE
        # -----------------------------------------------------
        st.markdown("---")
        st.markdown(
            '<div class="section-heading">Student Profile Summary</div>',
            unsafe_allow_html=True
        )

        def profile_item(label, value):
            return (
                '<div class="profile-item">'
                f'<div class="profile-label">{label}</div>'
                f'<div class="profile-value">{value}</div>'
                '</div>'
            )

        profile_col1, profile_col2, profile_col3 = st.columns(3)

        with profile_col1:
            st.markdown(
                profile_item("IQ", str(iq))
                + profile_item("CGPA", f"{cgpa:.1f} / 10")
                + profile_item(
                    "Previous Semester",
                    f"{prev_sem_result:.1f}"
                ),
                unsafe_allow_html=True
            )

        with profile_col2:
            st.markdown(
                profile_item(
                    "Academic Performance",
                    f"{academic_performance:.1f} / 10"
                )
                + profile_item("Internship", internship)
                + profile_item("Projects Completed", str(projects)),
                unsafe_allow_html=True
            )

        with profile_col3:
            st.markdown(
                profile_item(
                    "Communication",
                    f"{communication:.1f} / 10"
                )
                + profile_item(
                    "Extra-Curricular Score",
                    f"{extracurricular:.1f} / 10"
                )
                + profile_item("Prediction Model", "Random Forest"),
                unsafe_allow_html=True
            )

        # -----------------------------------------------------
        # DOWNLOAD PROFILE
        # -----------------------------------------------------
        report_data = pd.DataFrame([{
            **input_data.iloc[0].to_dict(),
            "Predicted_Placement": result_text,
            "Placement_Probability_Percent": (
                round(placed_probability, 2)
                if placed_probability is not None else None
            )
        }])

        st.download_button(
            "⬇️ Download Student Prediction (CSV)",
            data=report_data.to_csv(index=False).encode("utf-8"),
            file_name="student_placement_prediction.csv",
            mime="text/csv",
            use_container_width=True
        )

        st.caption(
            "This is a machine-learning estimate, not a guarantee "
            "of employment. Probability values depend on the saved "
            "model and its training data."
        )

    except Exception as error:
        st.error("Prediction failed. Check the model and input format.")
        st.exception(error)


# =========================================================
# FOOTER
# =========================================================
st.markdown("""
<div class="footer">
    <strong>Placement Intelligence</strong><br>
    AI-Based Student Placement Prediction System · Machine Learning Project
</div>
""", unsafe_allow_html=True)
