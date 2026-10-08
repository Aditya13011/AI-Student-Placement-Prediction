import streamlit as st
import pandas as pd
import joblib
from pathlib import Path


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Student Placement Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main {
        background-color: #0e1117;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        color: #b8b8b8;
        margin-bottom: 30px;
    }

    .result-box {
        padding: 30px;
        border-radius: 18px;
        text-align: center;
        margin-top: 20px;
        margin-bottom: 25px;
    }

    .placed-box {
        background-color: #102b1a;
        border: 2px solid #4ade80;
    }

    .not-placed-box {
        background-color: #351717;
        border: 2px solid #f87171;
    }

    .result-title {
        font-size: 32px;
        font-weight: 700;
        margin-bottom: 15px;
    }

    .result-description {
        font-size: 18px;
        margin-bottom: 15px;
    }

    .result-probability {
        font-size: 25px;
        font-weight: 600;
    }

    .profile-card {
        background-color: #171a21;
        border: 1px solid #30343d;
        border-radius: 15px;
        padding: 20px;
        margin-top: 10px;
        min-height: 170px;
    }

    .profile-item {
        margin-bottom: 14px;
        font-size: 16px;
    }

    .footer {
        text-align: center;
        color: #888888;
        margin-top: 50px;
        padding: 20px;
        border-top: 1px solid #30343d;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# FIND PROJECT FOLDER
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "final_random_forest_model.pkl"


# ============================================================
# LOAD RANDOM FOREST MODEL
# ============================================================

try:

    model = joblib.load(MODEL_PATH)

except Exception as e:

    st.error(
        f"❌ Error loading model: {e}"
    )

    st.info(
        "Make sure final_random_forest_model.pkl is in the same "
        "folder as app.py."
    )

    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <h2>📌 About the Project</h2>

        <p>
        This system predicts whether a student is likely to be
        placed based on academic performance, skills, projects
        and internship experience.
        </p>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    st.markdown("### ⚙️ Model")

    st.write("Random Forest Classifier")

    st.divider()

    st.markdown("### 📊 Dataset")

    st.write("College Student Placement Dataset")

    st.divider()

    st.markdown("### 🎯 Prediction")

    st.write("Placed / Not Placed")

    st.divider()

    st.caption("Machine Learning Mini Project")


# ============================================================
# MAIN TITLE
# ============================================================

st.markdown(
    '<div class="main-title">🎓 Student Placement Predictor</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
    AI-Based Student Placement Prediction System using Machine Learning
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# INFORMATION MESSAGE
# ============================================================

st.info(
    "Enter the student's academic and skill information "
    "to generate a placement prediction."
)


# ============================================================
# STUDENT INPUT SECTION
# ============================================================

st.markdown(
    "## 👨‍🎓 Enter Student Information"
)


# ============================================================
# ROW 1
# ============================================================

col1, col2, col3 = st.columns(3)


with col1:

    iq = st.number_input(
        "🧠 IQ",
        min_value=40,
        max_value=160,
        value=120,
        step=1
    )


with col2:

    prev_sem_result = st.number_input(
        "📚 Previous Semester Result",
        min_value=5.0,
        max_value=10.0,
        value=8.0,
        step=0.1
    )


with col3:

    cgpa = st.number_input(
        "🎓 CGPA",
        min_value=4.5,
        max_value=10.0,
        value=8.0,
        step=0.1
    )


# ============================================================
# ROW 2
# ============================================================

col1, col2, col3 = st.columns(3)


with col1:

    academic_performance = st.number_input(
        "📖 Academic Performance",
        min_value=1.0,
        max_value=10.0,
        value=8.0,
        step=0.1
    )


with col2:

    internship = st.selectbox(
        "💼 Internship Experience",
        ["Yes", "No"]
    )


with col3:

    extracurricular = st.number_input(
        "🏆 Extra-Curricular Score",
        min_value=1.0,
        max_value=10.0,
        value=7.0,
        step=0.1
    )


# ============================================================
# ROW 3
# ============================================================

col1, col2, col3 = st.columns(3)


with col1:

    communication = st.number_input(
        "🗣️ Communication Skills",
        min_value=1.0,
        max_value=10.0,
        value=8.0,
        step=0.1
    )


with col2:

    projects = st.number_input(
        "💻 Projects Completed",
        min_value=0,
        max_value=5,
        value=3,
        step=1
    )


with col3:

    st.write("")

    predict_button = st.button(
        "🔮 Predict Placement",
        use_container_width=True,
        type="primary"
    )


# ============================================================
# PREDICTION
# ============================================================

if predict_button:

    # --------------------------------------------------------
    # Convert internship value
    # --------------------------------------------------------

    if internship == "Yes":

        internship_value = 1

    else:

        internship_value = 0


    # --------------------------------------------------------
    # Create input DataFrame
    #
    # IMPORTANT:
    # These names MUST match the training dataset.
    # --------------------------------------------------------

    input_data = pd.DataFrame(
        {
            "IQ": [iq],
            "Prev_Sem_Result": [prev_sem_result],
            "CGPA": [cgpa],
            "Academic_Performance": [academic_performance],
            "Internship_Experience": [internship_value],
            "Extra_Curricular_Score": [extracurricular],
            "Communication_Skills": [communication],
            "Projects_Completed": [projects]
        }
    )


    # --------------------------------------------------------
    # MODEL PREDICTION
    # --------------------------------------------------------

    try:

        prediction = model.predict(input_data)[0]

        probabilities = model.predict_proba(input_data)[0]


        # ----------------------------------------------------
        # Determine probability classes
        # ----------------------------------------------------

        if hasattr(model, "classes_"):

            classes = list(model.classes_)

        else:

            classes = [0, 1]


        # ----------------------------------------------------
        # Find probability for Placed class
        # ----------------------------------------------------

        placed_probability = 0.0

        not_placed_probability = 0.0


        for index, class_value in enumerate(classes):

            if str(class_value).lower() in [
                "1",
                "yes",
                "placed"
            ]:

                placed_probability = probabilities[index] * 100

            else:

                not_placed_probability = probabilities[index] * 100


        # ----------------------------------------------------
        # If model uses numeric 0/1
        # ----------------------------------------------------

        if placed_probability == 0.0 and len(probabilities) == 2:

            if prediction == 1:

                placed_probability = probabilities[1] * 100

                not_placed_probability = probabilities[0] * 100

            else:

                placed_probability = probabilities[1] * 100

                not_placed_probability = probabilities[0] * 100


        # ----------------------------------------------------
        # Determine final prediction
        # ----------------------------------------------------

        prediction_text = str(prediction).lower()


        if prediction_text in [
            "1",
            "yes",
            "placed",
            "true"
        ]:

            result = "PLACED"

            result_icon = "✅"

            description = (
                "The model predicts that the student is "
                "likely to be placed."
            )

            result_class = "placed-box"


        else:

            result = "NOT PLACED"

            result_icon = "❌"

            description = (
                "The model predicts that the student is "
                "unlikely to be placed."
            )

            result_class = "not-placed-box"


        # ====================================================
        # PREDICTION RESULT
        # ====================================================

        st.markdown(
            "## 📊 Prediction Result"
        )


        # ----------------------------------------------------
        # FIXED RESULT HTML
        #
        # IMPORTANT:
        # The HTML is deliberately kept on single lines
        # inside the f-string to prevent Streamlit from
        # displaying the HTML tags as text.
        # ----------------------------------------------------

        result_html = (
            f'<div class="result-box {result_class}">'
            f'<div class="result-title">'
            f'{result_icon} {result}'
            f'</div>'
            f'<div class="result-description">'
            f'{description}'
            f'</div>'
            f'<div class="result-probability">'
            f'Placement Probability: '
            f'{placed_probability:.2f}%'
            f'</div>'
            f'</div>'
        )


        st.markdown(
            result_html,
            unsafe_allow_html=True
        )


        # ====================================================
        # PROBABILITY SECTION
        # ====================================================

        probability_col1, probability_col2 = st.columns(2)


        with probability_col1:

            st.markdown(
                "### ✅ Placement Probability"
            )

            st.metric(
                label="Probability",
                value=f"{placed_probability:.2f}%"
            )


        with probability_col2:

            st.markdown(
                "### ❌ Not Placed Probability"
            )

            st.metric(
                label="Probability",
                value=f"{not_placed_probability:.2f}%"
            )


        # ====================================================
        # PROGRESS BAR
        # ====================================================

        st.markdown(
            "### 📈 Placement Probability"
        )

        progress_value = placed_probability / 100

        progress_value = max(
            0.0,
            min(1.0, progress_value)
        )

        st.progress(
            progress_value
        )


        # ====================================================
        # STUDENT PROFILE SUMMARY
        # ====================================================

        st.markdown(
            "## 📋 Student Profile Summary"
        )


        profile_col1, profile_col2, profile_col3 = st.columns(3)


        # ----------------------------------------------------
        # PROFILE CARD 1
        # ----------------------------------------------------

        with profile_col1:

            st.markdown(
                f"""
                <div class="profile-card">

                <div class="profile-item">
                🧠 <b>IQ:</b> {iq}
                </div>

                <div class="profile-item">
                🎓 <b>CGPA:</b> {cgpa:.1f}
                </div>

                <div class="profile-item">
                📚 <b>Previous Semester:</b>
                {prev_sem_result:.1f}
                </div>

                </div>
                """,
                unsafe_allow_html=True
            )


        # ----------------------------------------------------
        # PROFILE CARD 2
        # ----------------------------------------------------

        with profile_col2:

            st.markdown(
                f"""
                <div class="profile-card">

                <div class="profile-item">
                📖 <b>Academic Performance:</b>
                {academic_performance:.1f}
                </div>

                <div class="profile-item">
                💼 <b>Internship:</b>
                {internship}
                </div>

                <div class="profile-item">
                💻 <b>Projects:</b>
                {projects}
                </div>

                </div>
                """,
                unsafe_allow_html=True
            )


        # ----------------------------------------------------
        # PROFILE CARD 3
        # ----------------------------------------------------

        with profile_col3:

            st.markdown(
                f"""
                <div class="profile-card">

                <div class="profile-item">
                🗣️ <b>Communication:</b>
                {communication:.1f}
                </div>

                <div class="profile-item">
                🏆 <b>Extra-Curricular:</b>
                {extracurricular:.1f}
                </div>

                <div class="profile-item">
                🤖 <b>Model:</b>
                Random Forest
                </div>

                </div>
                """,
                unsafe_allow_html=True
            )


        # ====================================================
        # MODEL INFORMATION
        # ====================================================

        st.markdown(
            "## 🤖 Model Information"
        )

        st.success(
            "Prediction generated successfully using "
            "the Random Forest Classifier."
        )


    # ========================================================
    # ERROR HANDLING
    # ========================================================

    except Exception as e:

        st.error(
            "❌ Prediction failed."
        )

        st.exception(e)


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
    AI-Based Student Placement Prediction System |
    Machine Learning Mini Project
    </div>
    """,
    unsafe_allow_html=True
)