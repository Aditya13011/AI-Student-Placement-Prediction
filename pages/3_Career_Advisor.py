
import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Student Career Advisor",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 AI Student Career Advisor")
st.caption(
    "Analyze your current profile and generate a personalized "
    "placement preparation plan."
)
st.divider()

st.info(
    "This advisor uses transparent, rule-based recommendations. "
    "It is a preparation tool, not a guarantee of placement."
)

# --------------------------------------------------
# STUDENT PROFILE
# --------------------------------------------------
st.header("👨‍🎓 Enter Your Profile")

col1, col2 = st.columns(2)

with col1:
    cgpa = st.slider(
        "CGPA",
        min_value=4.5,
        max_value=10.0,
        value=7.5,
        step=0.1
    )

    communication = st.slider(
        "Communication Skills",
        min_value=1,
        max_value=10,
        value=6
    )

    projects = st.number_input(
        "Completed Projects",
        min_value=0,
        max_value=20,
        value=2
    )

with col2:
    internship = st.selectbox(
        "Internship Experience",
        ["No", "Yes"]
    )

    extracurricular = st.slider(
        "Extra-Curricular Score",
        min_value=1,
        max_value=10,
        value=6
    )

    target_role = st.selectbox(
        "Target Career Role",
        [
            "Software Developer",
            "Data Analyst",
            "Machine Learning Engineer",
            "Android Developer",
            "Backend Developer"
        ]
    )

st.divider()

# --------------------------------------------------
# ROLE-SPECIFIC SKILL RECOMMENDATIONS
# --------------------------------------------------
role_skills = {
    "Software Developer": [
        "Java or Python fundamentals",
        "Data Structures and Algorithms",
        "Object-Oriented Programming",
        "SQL and database fundamentals",
        "Git and GitHub"
    ],
    "Data Analyst": [
        "Python and Pandas",
        "SQL queries",
        "Excel and data cleaning",
        "Data visualization",
        "Statistics fundamentals"
    ],
    "Machine Learning Engineer": [
        "Python and NumPy",
        "Pandas and data preprocessing",
        "Machine learning algorithms",
        "Model evaluation",
        "Feature engineering"
    ],
    "Android Developer": [
        "Java or Kotlin",
        "Android Studio",
        "XML or Jetpack Compose",
        "SQLite or Room",
        "REST API integration"
    ],
    "Backend Developer": [
        "Java fundamentals",
        "Spring Boot",
        "REST API development",
        "SQL and database design",
        "Authentication and testing"
    ]
}

# --------------------------------------------------
# GENERATE RECOMMENDATIONS
# --------------------------------------------------
if st.button(
    "🚀 Generate My Career Plan",
    type="primary",
    use_container_width=True
):

    recommendations = []
    strengths = []

    # Academic guidance
    if cgpa < 7.0:
        recommendations.append(
            "Academic improvement: revise difficult subjects, "
            "practice previous papers, and improve consistency."
        )
    elif cgpa < 8.0:
        strengths.append(
            "Your CGPA provides a foundation to build on. "
            "Continue improving subject fundamentals."
        )
    else:
        strengths.append(
            "Your CGPA is a strength. Maintain it while "
            "developing practical skills."
        )

    # Communication guidance
    if communication < 6:
        recommendations.append(
            "Communication: practise speaking for 15 minutes daily, "
            "explain technical concepts aloud, and complete mock interviews."
        )
    else:
        strengths.append(
            "Your communication score is a useful starting point. "
            "Continue practising technical explanations."
        )

    # Project guidance
    if projects < 2:
        recommendations.append(
            "Projects: build and document at least two complete "
            "projects, including a README and screenshots."
        )
    elif projects < 4:
        recommendations.append(
            "Projects: improve your existing projects with testing, "
            "documentation, and a live demonstration."
        )
    else:
        strengths.append(
            "You have several completed projects. Focus on quality, "
            "clear documentation, and explaining your design decisions."
        )

    # Internship guidance
    if internship == "No":
        recommendations.append(
            "Experience: apply for internships, open-source work, "
            "college research projects, or practical team projects."
        )
    else:
        strengths.append(
            "You report internship experience. Prepare examples of "
            "your contributions and what you learned."
        )

    # Extracurricular guidance
    if extracurricular < 5:
        recommendations.append(
            "Teamwork: participate in a hackathon, technical club, "
            "group project, or presentation."
        )

    # Role-specific recommendations
    skills = role_skills[target_role]

    st.header("📋 Your Career Readiness Summary")

    c1, c2, c3 = st.columns(3)
    c1.metric("Target Role", target_role)
    c2.metric("CGPA", f"{cgpa:.1f}/10")
    c3.metric("Projects", int(projects))

    st.subheader("💪 Current Strengths")

    if strengths:
        for strength in strengths:
            st.success(strength)
    else:
        st.write(
            "You have areas to develop. Use the recommendations below "
            "as your starting point."
        )

    st.subheader("🎯 Priority Improvement Areas")

    if recommendations:
        for index, recommendation in enumerate(
            recommendations, start=1
        ):
            st.write(f"**{index}.** {recommendation}")
    else:
        st.success(
            "No priority areas were flagged by these simple rules. "
            "Continue practising and seek feedback."
        )

    st.subheader(f"🛠️ Skills for {target_role}")

    for skill in skills:
        st.checkbox(
            skill,
            key=f"career_skill_{target_role}_{skill}"
        )

    st.subheader("📅 Suggested 4-Week Preparation Plan")

    plan = pd.DataFrame({
        "Week": [
            "Week 1",
            "Week 2",
            "Week 3",
            "Week 4"
        ],
        "Focus": [
            "Review fundamentals and identify weak areas",
            "Practise role-specific skills and coding exercises",
            "Build or improve a practical project",
            "Revise your resume and practise mock interviews"
        ],
        "Expected Output": [
            "Personal skill-gap checklist",
            "Practice solutions and notes",
            "Documented project milestone",
            "Resume draft and interview feedback"
        ]
    })

    st.dataframe(
        plan,
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        "⬇️ Download Career Plan",
        data=plan.to_csv(index=False).encode("utf-8"),
        file_name="student_career_plan.csv",
        mime="text/csv"
    )

    st.caption(
        "Recommendations are based on manually defined rules, "
        "not on a trained career recommendation model."
    )

st.divider()
st.caption(
    "Student Placement Intelligence System | Career Advisor"
)
