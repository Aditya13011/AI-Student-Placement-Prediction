
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
from io import BytesIO
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle
)

st.set_page_config(
    page_title="Student Reports",
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
    max-width: 1250px;
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
.report-hero {
    background: linear-gradient(120deg, #182443, #25204a);
    border: 1px solid #35436b;
    padding: 28px 32px;
    border-radius: 22px;
    margin-bottom: 24px;
}
.report-hero h1 {
    font-size: 32px;
    font-weight: 800;
    margin: 0;
}
.report-hero p {
    color: #b8c5e2;
    margin-bottom: 0;
}
.report-card {
    background: #111a2d;
    border: 1px solid #293653;
    padding: 22px;
    border-radius: 16px;
    margin-bottom: 16px;
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
<div class="report-hero">
    <h1>📄 Student Reports Centre</h1>
    <p>Create a downloadable PDF summary of an individual student's profile and model prediction.</p>
</div>
""", unsafe_allow_html=True)

st.info(
    "The report contains a model-generated estimate, not a guarantee of placement. "
    "Enter the same feature values and encodings used by your trained model."
)


@st.cache_resource
def load_model(path):
    return joblib.load(path)


def make_pdf(student_name, values, prediction, probability):
    buffer = BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=0.65 * inch,
        leftMargin=0.65 * inch,
        topMargin=0.65 * inch,
        bottomMargin=0.65 * inch
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=26,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#24366b"),
        spaceAfter=12
    )

    heading_style = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=12,
        textColor=colors.HexColor("#3347a5"),
        spaceBefore=12,
        spaceAfter=8
    )

    normal_style = styles["BodyText"]
    normal_style.leading = 16

    story = [
        Paragraph("STUDENT PLACEMENT REPORT", title_style),
        Paragraph(
            f"Student: {student_name}",
            normal_style
        ),
        Paragraph(
            f"Generated: {datetime.now().strftime('%d %B %Y, %I:%M %p')}",
            normal_style
        ),
        Spacer(1, 16),
        Paragraph("Student Profile", heading_style)
    ]

    profile_data = [["Feature", "Entered Value"]]
    for feature in FEATURES:
        profile_data.append([feature, str(values[feature])])

    profile_table = Table(
        profile_data,
        colWidths=[3.3 * inch, 2.6 * inch],
        repeatRows=1
    )

    profile_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#24366b")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#d6dbea")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [
            colors.white, colors.HexColor("#f2f5fc")
        ]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 9),
        ("RIGHTPADDING", (0, 0), (-1, -1), 9),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7)
    ]))

    story.extend([
        profile_table,
        Spacer(1, 18),
        Paragraph("Prediction Summary", heading_style),
        Paragraph(f"Predicted Outcome: <b>{prediction}</b>", normal_style)
    ])

    if probability is not None:
        story.append(
            Paragraph(
                f"Model probability for the predicted positive class: "
                f"<b>{probability:.2f}%</b>",
                normal_style
            )
        )

    story.extend([
        Spacer(1, 18),
        Paragraph("Important Disclaimer", heading_style),
        Paragraph(
            "This report is generated by a machine learning model for educational "
            "and informational purposes. Its predictions may be inaccurate and "
            "should not be treated as a guarantee of placement or as a substitute "
            "for an actual recruitment decision.",
            normal_style
        )
    ])

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


if not MODEL_PATH.exists():
    st.error(f"Model file not found: {MODEL_PATH}")
    st.info("Place final_random_forest_model.pkl in your project root folder.")
    st.stop()

try:
    model = load_model(str(MODEL_PATH))
except Exception as exc:
    st.error(f"Could not load the saved model: {exc}")
    st.stop()

st.markdown(
    '<div class="report-card"><strong>Step 1 — Student Information</strong></div>',
    unsafe_allow_html=True
)

student_name = st.text_input(
    "Student name or identifier",
    value="Student"
)

col1, col2 = st.columns(2)

with col1:
    iq = st.number_input("IQ", min_value=0.0, max_value=300.0, value=100.0)
    prev_result = st.number_input(
        "Previous Semester Result",
        min_value=0.0, max_value=100.0, value=70.0
    )
    cgpa = st.number_input(
        "CGPA", min_value=0.0, max_value=10.0, value=7.0, step=0.1
    )
    academic = st.number_input(
        "Academic Performance",
        min_value=0.0, max_value=10.0, value=7.0
    )

with col2:
    internship = st.selectbox(
        "Internship Experience",
        ["No", "Yes"]
    )
    extracurricular = st.number_input(
        "Extra-Curricular Score",
        min_value=0.0, max_value=10.0, value=6.0
    )
    communication = st.number_input(
        "Communication Skills",
        min_value=0.0, max_value=10.0, value=6.0
    )
    projects = st.number_input(
        "Projects Completed",
        min_value=0, max_value=100, value=2, step=1
    )

if st.button("Generate Student Report", use_container_width=True):
    values = {
        "IQ": iq,
        "Prev_Sem_Result": prev_result,
        "CGPA": cgpa,
        "Academic_Performance": academic,
        "Internship_Experience": 1 if internship == "Yes" else 0,
        "Extra_Curricular_Score": extracurricular,
        "Communication_Skills": communication,
        "Projects_Completed": projects
    }

    try:
        input_df = pd.DataFrame(
            [[values[feature] for feature in FEATURES]],
            columns=FEATURES
        )

        if hasattr(model, "n_features_in_"):
            if model.n_features_in_ != len(FEATURES):
                st.error(
                    "The saved model does not expect eight features. "
                    "Verify your training pipeline before generating reports."
                )
                st.stop()

        prediction_value = model.predict(input_df)[0]
        prediction_text = (
            "Placed"
            if str(prediction_value).strip().lower()
            in ["1", "1.0", "true", "yes", "placed"]
            else "Not Placed"
        )

        probability = None
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_df)[0]
            classes = list(getattr(model, "classes_", []))
            positive_index = next(
                (
                    i for i, label in enumerate(classes)
                    if str(label).strip().lower()
                    in ["1", "1.0", "true", "yes", "placed"]
                ),
                None
            )
            if positive_index is not None:
                probability = float(probabilities[positive_index] * 100)

        pdf_bytes = make_pdf(
            student_name.strip() or "Student",
            values,
            prediction_text,
            probability
        )

        st.session_state["student_report"] = {
            "name": student_name.strip() or "Student",
            "prediction": prediction_text,
            "probability": probability,
            "pdf": pdf_bytes
        }

    except Exception as exc:
        st.error(f"Could not generate the report: {exc}")
        st.info(
            "Verify that the saved model accepts these eight numerical features "
            "in the expected order."
        )

if "student_report" in st.session_state:
    report = st.session_state["student_report"]

    st.divider()
    st.subheader("Report Preview")

    a, b = st.columns(2)
    a.metric("Student", report["name"])
    b.metric("Predicted Outcome", report["prediction"])

    if report["probability"] is not None:
        st.metric(
            "Model Probability for Placed Class",
            f"{report['probability']:.2f}%"
        )

    st.markdown(
        '<div class="report-card">Your PDF report is ready. Download it below.</div>',
        unsafe_allow_html=True
    )

    st.download_button(
        "⬇️ Download PDF Report",
        data=report["pdf"],
        file_name="student_placement_report.pdf",
        mime="application/pdf",
        use_container_width=True
    )

st.markdown("---")
st.markdown(
    '<p class="small-note">Student Reports · Keep downloaded reports secure because they may contain student information.</p>',
    unsafe_allow_html=True
)
