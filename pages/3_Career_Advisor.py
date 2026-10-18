
import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Career Advisor",
    page_icon="🎯",
    layout="wide"
)

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
div[data-testid="stMetric"] {
    background: #151e33;
    border: 1px solid #293653;
    border-radius: 16px;
    padding: 18px;
}
div[data-testid="stMetricValue"] {
    color: #8b9cff;
}
.advisor-hero {
    background: linear-gradient(120deg, #182443, #25204a);
    border: 1px solid #35436b;
    padding: 28px 32px;
    border-radius: 22px;
    margin-bottom: 24px;
}
.advisor-hero h1 {
    margin: 0;
    font-size: 32px;
    font-weight: 800;
}
.advisor-hero p {
    color: #b8c5e2;
    margin-bottom: 0;
}
.advisor-card {
    background: #111a2d;
    border: 1px solid #293653;
    padding: 20px;
    border-radius: 16px;
    margin-bottom: 12px;
}
.advisor-card p {
    color: #bac7e1;
}
div.stButton > button {
    background: linear-gradient(90deg, #697bff, #8b5cf6);
    color: white;
    border: 0;
    border-radius: 10px;
    padding: 0.65rem 1.1rem;
    font-weight: 700;
}
div.stButton > button:hover {
    border: 1px solid #aab4ff;
    color: white;
}
.small-note {
    color: #aab8d3;
    font-size: 13px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="advisor-hero">
    <h1>🎯 Career Advisor</h1>
    <p>Explore career paths and identify practical skills to strengthen your placement readiness.</p>
</div>
""", unsafe_allow_html=True)

st.info(
    "This page provides rule-based guidance from the details you enter. "
    "It is not a guarantee of hiring outcomes."
)

st.subheader("Your Student Profile")

col1, col2 = st.columns(2)

with col1:
    cgpa = st.number_input(
        "CGPA",
        min_value=0.0,
        max_value=10.0,
        value=7.0,
        step=0.1
    )
    communication = st.slider(
        "Communication Skills",
        min_value=0,
        max_value=10,
        value=6
    )
    academic = st.slider(
        "Academic Performance",
        min_value=0,
        max_value=10,
        value=7
    )
    internship = st.selectbox(
        "Internship Experience",
        ["No", "Yes"]
    )

with col2:
    projects = st.number_input(
        "Projects Completed",
        min_value=0,
        max_value=100,
        value=2,
        step=1
    )
    extracurricular = st.slider(
        "Extra-Curricular Score",
        min_value=0,
        max_value=10,
        value=6
    )
    previous_result = st.number_input(
        "Previous Semester Result",
        min_value=0.0,
        max_value=100.0,
        value=70.0,
        step=0.5
    )
    career = st.selectbox(
        "Career Interest",
        [
            "Software Developer",
            "Data Analyst",
            "Machine Learning Engineer",
            "Android Developer",
            "QA / Testing Engineer"
        ]
    )

if st.button("Generate Career Plan", use_container_width=True):
    st.session_state["career_profile"] = {
        "cgpa": cgpa,
        "communication": communication,
        "academic": academic,
        "internship": internship,
        "projects": int(projects),
        "extracurricular": extracurricular,
        "previous_result": previous_result,
        "career": career
    }

if "career_profile" in st.session_state:
    profile = st.session_state["career_profile"]

    cgpa = profile["cgpa"]
    communication = profile["communication"]
    academic = profile["academic"]
    internship = profile["internship"]
    projects = profile["projects"]
    extracurricular = profile["extracurricular"]
    previous_result = profile["previous_result"]
    career = profile["career"]

    st.divider()
    st.subheader("Your Career Readiness Snapshot")

    readiness_points = [
        min(cgpa / 10 * 100, 100),
        communication * 10,
        academic * 10,
        min(projects / 5 * 100, 100),
        100 if internship == "Yes" else 0,
        extracurricular * 10
    ]
    readiness = round(sum(readiness_points) / len(readiness_points))

    m1, m2, m3 = st.columns(3)
    m1.metric("Readiness Indicator", f"{readiness}%")
    m2.metric("Projects Completed", projects)
    m3.metric("Internship", internship)

    st.progress(readiness / 100)
    st.caption(
        "This is a simple planning indicator, not a validated hiring probability."
    )

    recommendations = []

    if cgpa < 7.0:
        recommendations.append((
            "Improve academic consistency",
            "Review core subjects regularly, practise previous questions, and work toward improving semester results."
        ))

    if communication < 7:
        recommendations.append((
            "Practise communication",
            "Spend 15–20 minutes daily on explaining technical concepts, mock interviews, and speaking practice."
        ))

    if academic < 7:
        recommendations.append((
            "Strengthen technical fundamentals",
            "Identify weak subjects and solve practical problems to improve conceptual understanding."
        ))

    if projects < 2:
        recommendations.append((
            "Build portfolio projects",
            "Complete at least one well-documented project and explain its problem, architecture, and implementation."
        ))

    if internship == "No":
        recommendations.append((
            "Gain practical experience",
            "Look for internships, open-source contributions, virtual experience programs, or relevant practical work."
        ))

    if extracurricular < 5:
        recommendations.append((
            "Improve teamwork and collaboration",
            "Participate in group projects, technical clubs, hackathons, or peer-learning activities."
        ))

    role_skills = {
        "Software Developer": [
            "Java or Python fundamentals",
            "DSA and problem solving",
            "OOP, SQL, Git, and project explanation"
        ],
        "Data Analyst": [
            "SQL and data cleaning",
            "Python with Pandas",
            "Statistics and data visualisation"
        ],
        "Machine Learning Engineer": [
            "Python and data preprocessing",
            "Machine learning algorithms and evaluation",
            "Model deployment and project documentation"
        ],
        "Android Developer": [
            "Java or Kotlin",
            "Android Studio and app lifecycle",
            "SQLite or Room and API integration"
        ],
        "QA / Testing Engineer": [
            "Software testing fundamentals",
            "Test cases, bug reports, and SQL",
            "Automation testing basics"
        ]
    }

    st.subheader("Personalised Improvement Plan")

    if recommendations:
        for index, (title, description) in enumerate(recommendations, start=1):
            st.markdown(
                f"""
                <div class="advisor-card">
                    <strong>{index}. {title}</strong>
                    <p>{description}</p>
                </div>
                """,
                unsafe_allow_html=True
            )
    else:
        st.success(
            "Your entered profile meets the basic thresholds used by this advisor. "
            "Continue practising interviews and improving your projects."
        )

    st.subheader(f"Skills to Focus On: {career}")

    for skill in role_skills[career]:
        st.markdown(f"- {skill}")

    plan = pd.DataFrame({
        "Career Interest": [career] * len(role_skills[career]),
        "Recommended Focus Area": role_skills[career]
    })

    st.download_button(
        "⬇️ Download Career Plan",
        data=plan.to_csv(index=False).encode("utf-8"),
        file_name="career_development_plan.csv",
        mime="text/csv",
        use_container_width=True
    )

st.markdown("---")
st.markdown(
    '<p class="small-note">Career Advisor · Recommendations are educational guidance and do not guarantee placement.</p>',
    unsafe_allow_html=True
)
