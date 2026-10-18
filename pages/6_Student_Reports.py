
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
from io import BytesIO
from datetime import datetime
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle
)

st.set_page_config(
    page_title="Student PDF Reports",
    page_icon="📄",
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
    "Projects_Completed"
]


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def create_pdf(student, prediction, probability, recommendations):
    buffer = BytesIO()

    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=0.65 * inch,
        leftMargin=0.65 * inch,
        topMargin=0.65 * inch,
        bottomMargin=0.65 * inch
    )

    styles = getSampleStyleSheet()
    story = []

    story.append(
        Paragraph("Student Placement Assessment Report", styles["Title"])
    )
    story.append(Spacer(1, 8))
    story.append(
        Paragraph(
            f"Generated on: {datetime.now().strftime('%d %B %Y, %I:%M %p')}",
            styles["Normal"]
        )
    )
    story.append(Spacer(1, 18))

    story.append(Paragraph("Student Profile", styles["Heading2"]))

    profile_data = [
        ["Parameter", "Student Value"],
        ["IQ", str(student["IQ"])],
        ["Previous Semester Result", str(student["Prev_Sem_Result"])],
        ["CGPA", str(student["CGPA"])],
        ["Academic Performance", str(student["Academic_Performance"])],
        [
            "Internship Experience",
            "Yes" if student["Internship_Experience"] == 1 else "No"
        ],
        ["Extra-Curricular Score", str(student["Extra_Curricular_Score"])],
        ["Communication Skills", str(student["Communication_Skills"])],
        ["Projects Completed", str(student["Projects_Completed"])]
    ]

    profile_table = Table(profile_data, colWidths=[3 * inch, 3.2 * inch])
    profile_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#243B64")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("PADDING", (0, 0), (-1, -1), 7),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE")
    ]))

    story.append(profile_table)
    story.append(Spacer(1, 20))

    story.append(Paragraph("Model Prediction", styles["Heading2"]))
    story.append(Paragraph(
        f"Predicted Outcome: <b>{prediction}</b>",
        styles["Normal"]
    ))

    if probability is not None:
        story.append(Paragraph(
            f"Model-estimated probability of placement: {probability:.2f}%",
            styles["Normal"]
        ))

    story.append(Spacer(1, 15))
    story.append(Paragraph("Preparation Recommendations", styles["Heading2"]))

    for item in recommendations:
        story.append(Paragraph(f"• {item}", styles["Normal"]))
        story.append(Spacer(1, 6))

    story.append(Spacer(1, 15))
    story.append(Paragraph("Important Disclaimer", styles["Heading2"]))
    story.append(Paragraph(
        "This report contains a machine-learning model prediction, "
        "not a guarantee of employment or placement. Results depend on "
        "the model and the quality of the input data. Use this report "
        "for educational guidance only.",
        styles["Normal"]
    ))

    document.build(story)
    buffer.seek(0)
    return buffer.getvalue()


st.title("📄 Student PDF Reports")
st.write(
    "Create a downloadable placement assessment report "
    "for an individual student."
)

if not MODEL_PATH.exists():
    st.error(
        "Model file not found. Ensure "
        "'final_random_forest_model.pkl' is in the project root."
    )
    st.stop()

model = load_model()

st.subheader("Enter Student Details")

with st.form("student_report_form"):
    col1, col2 = st.columns(2)

    with col1:
        iq = st.number_input("IQ", 50, 200, 110)
        previous_result = st.number_input(
            "Previous Semester Result", 0.0, 100.0, 75.0
        )
        cgpa = st.number_input("CGPA", 0.0, 10.0, 8.0)
        academic = st.number_input(
            "Academic Performance", 0.0, 10.0, 8.0
        )

    with col2:
        internship_text = st.selectbox(
            "Internship Experience", ["Yes", "No"]
        )
        extracurricular = st.number_input(
            "Extra-Curricular Score", 0.0, 10.0, 7.0
        )
        communication = st.number_input(
            "Communication Skills", 0.0, 10.0, 8.0
        )
        projects = st.number_input(
            "Projects Completed", 0, 20, 3
        )

    generate = st.form_submit_button(
        "Generate PDF Report",
        type="primary",
        use_container_width=True
    )

if generate:
    student = {
        "IQ": iq,
        "Prev_Sem_Result": previous_result,
        "CGPA": cgpa,
        "Academic_Performance": academic,
        "Internship_Experience": 1 if internship_text == "Yes" else 0,
        "Extra_Curricular_Score": extracurricular,
        "Communication_Skills": communication,
        "Projects_Completed": projects
    }

    input_df = pd.DataFrame([student], columns=FEATURES)

    try:
        if (
            hasattr(model, "n_features_in_")
            and model.n_features_in_ != len(FEATURES)
        ):
            st.error(
                "The model's expected feature count does not match "
                "this report form. Verify the model training pipeline."
            )
            st.stop()

        prediction_value = model.predict(input_df)[0]

        if str(prediction_value).strip().lower() in ["1", "yes", "true"]:
            prediction = "Placed"
        else:
            prediction = "Not Placed"

        probability = None

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_df)[0]
            classes = list(model.classes_)

            yes_index = next(
                (
                    i for i, value in enumerate(classes)
                    if str(value).strip().lower() in ["1", "yes", "true"]
                ),
                None
            )

            if yes_index is not None:
                probability = float(probabilities[yes_index] * 100)

        recommendations = []

        if cgpa < 7.0:
            recommendations.append(
                "Review academic fundamentals and strengthen subject knowledge."
            )

        if communication < 6:
            recommendations.append(
                "Practise communication, presentations, and mock interviews."
            )

        if projects < 2:
            recommendations.append(
                "Build at least two practical projects and document them."
            )

        if internship_text == "No":
            recommendations.append(
                "Explore internships, open-source contributions, and team projects."
            )

        if not recommendations:
            recommendations.append(
                "Continue practising technical skills and prepare for interviews."
            )

        pdf_bytes = create_pdf(
            student,
            prediction,
            probability,
            recommendations
        )

        st.subheader("Report Preview")
        st.metric("Predicted Outcome", prediction)

        if probability is not None:
            st.metric("Model-Estimated Placement Probability", f"{probability:.2f}%")

        st.success("Your PDF report is ready.")

        st.download_button(
            "⬇️ Download Student Report (PDF)",
            data=pdf_bytes,
            file_name="student_placement_report.pdf",
            mime="application/pdf",
            use_container_width=True
        )

    except Exception as error:
        st.error(
            "Could not generate the report. The saved model may require "
            "a different feature format or order."
        )
        st.code(str(error))

st.divider()
st.caption("Student Placement Intelligence System | PDF Reports")
