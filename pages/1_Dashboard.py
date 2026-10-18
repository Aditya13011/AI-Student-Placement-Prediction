
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(
    page_title="Placement Dashboard",
    page_icon="📊",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "college_student_placement_dataset.csv"

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
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1450px;
}
[data-testid="stSidebar"] {
    background: #0b1020;
    border-right: 1px solid #27314a;
}
h1, h2, h3 {
    color: #f1f5ff !important;
}
[data-testid="stMetric"] {
    background: #151e33;
    border: 1px solid #293653;
    border-radius: 16px;
    padding: 20px;
}
[data-testid="stMetricLabel"] {
    color: #aab8d3 !important;
}
[data-testid="stMetricValue"] {
    color: #8b9cff !important;
}
div[data-testid="stDataFrame"] {
    border: 1px solid #293653;
    border-radius: 12px;
    overflow: hidden;
}
.dashboard-hero {
    background: linear-gradient(120deg, #182443, #25204a);
    border: 1px solid #35436b;
    padding: 28px 32px;
    border-radius: 22px;
    margin-bottom: 25px;
}
.dashboard-hero h1 {
    margin: 0;
    font-size: 32px;
    font-weight: 800;
}
.dashboard-hero p {
    color: #b8c5e2;
    margin-top: 10px;
    margin-bottom: 0;
}
.section-card {
    background: #111a2d;
    border: 1px solid #293653;
    padding: 18px;
    border-radius: 16px;
    margin-bottom: 14px;
}
.small-note {
    color: #aab8d3;
    font-size: 13px;
}
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data(path):
    return pd.read_csv(path)


st.markdown("""
<div class="dashboard-hero">
    <h1>📊 Placement Intelligence Dashboard</h1>
    <p>Explore student performance, placement outcomes, and academic trends.</p>
</div>
""", unsafe_allow_html=True)

if not DATA_PATH.exists():
    st.error(f"Dataset not found: {DATA_PATH}")
    st.info("Keep college_student_placement_dataset.csv in the project root folder.")
    st.stop()

try:
    df = load_data(str(DATA_PATH))
except Exception as exc:
    st.error(f"Could not load the dataset: {exc}")
    st.stop()

required_columns = [
    "IQ", "Prev_Sem_Result", "CGPA", "Academic_Performance",
    "Internship_Experience", "Extra_Curricular_Score",
    "Communication_Skills", "Projects_Completed", "Placement"
]

missing = [column for column in required_columns if column not in df.columns]
if missing:
    st.error(f"Missing required dataset columns: {', '.join(missing)}")
    st.stop()

df = df.copy()
df["Placement_Label"] = (
    df["Placement"].astype(str).str.strip().str.lower()
    .map({"1": "Placed", "1.0": "Placed", "yes": "Placed",
          "true": "Placed", "0": "Not Placed", "0.0": "Not Placed",
          "no": "Not Placed", "false": "Not Placed"})
    .fillna(df["Placement"].astype(str))
)

numeric_columns = [
    "IQ", "Prev_Sem_Result", "CGPA", "Academic_Performance",
    "Extra_Curricular_Score", "Communication_Skills",
    "Projects_Completed"
]
for column in numeric_columns:
    df[column] = pd.to_numeric(df[column], errors="coerce")

df = df.dropna(subset=numeric_columns + ["Placement_Label"])

st.sidebar.header("Dashboard Filters")
st.sidebar.caption("Filters apply to the charts and summary below.")

cgpa_min = float(df["CGPA"].min())
cgpa_max = float(df["CGPA"].max())

if cgpa_min < cgpa_max:
    cgpa_range = st.sidebar.slider(
        "CGPA range",
        min_value=cgpa_min,
        max_value=cgpa_max,
        value=(cgpa_min, cgpa_max)
    )
else:
    cgpa_range = (cgpa_min, cgpa_max)

placement_options = sorted(df["Placement_Label"].unique().tolist())
selected_placements = st.sidebar.multiselect(
    "Placement status",
    options=placement_options,
    default=placement_options
)

filtered = df[
    df["CGPA"].between(cgpa_range[0], cgpa_range[1])
    & df["Placement_Label"].isin(selected_placements)
].copy()

if filtered.empty:
    st.warning("No students match these filters. Adjust the filters in the sidebar.")
    st.stop()

total_students = len(filtered)
placed_students = int((filtered["Placement_Label"] == "Placed").sum())
not_placed = total_students - placed_students
placement_rate = placed_students / total_students * 100 if total_students else 0
avg_cgpa = filtered["CGPA"].mean()

m1, m2, m3, m4 = st.columns(4)
m1.metric("Students Analysed", f"{total_students:,}")
m2.metric("Placed Students", f"{placed_students:,}")
m3.metric("Placement Rate", f"{placement_rate:.1f}%")
m4.metric("Average CGPA", f"{avg_cgpa:.2f}")

st.divider()

left, right = st.columns(2)

with left:
    st.subheader("Placement Distribution")
    placement_counts = (
        filtered["Placement_Label"]
        .value_counts()
        .rename_axis("Status")
        .reset_index(name="Students")
    )
    fig = px.pie(
        placement_counts,
        names="Status",
        values="Students",
        hole=0.62,
        color="Status",
        color_discrete_map={
            "Placed": "#7c8cff",
            "Not Placed": "#f59eae"
        }
    )
    fig.update_traces(
        textposition="inside",
        textinfo="percent+label",
        marker=dict(line=dict(color="#111a2d", width=3))
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e7edff",
        legend_title_text="",
        margin=dict(t=20, b=20, l=10, r=10)
    )
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("CGPA vs Placement")
    fig = px.box(
        filtered,
        x="Placement_Label",
        y="CGPA",
        color="Placement_Label",
        color_discrete_map={
            "Placed": "#7c8cff",
            "Not Placed": "#f59eae"
        },
        points="outliers"
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e7edff",
        xaxis_title="Placement Status",
        yaxis_title="CGPA",
        showlegend=False,
        margin=dict(t=20, b=20, l=10, r=10)
    )
    fig.update_xaxes(gridcolor="#283550")
    fig.update_yaxes(gridcolor="#283550")
    st.plotly_chart(fig, use_container_width=True)

left, right = st.columns(2)

with left:
    st.subheader("Academic Performance")
    academic = (
        filtered.groupby("Placement_Label", as_index=False)
        ["Academic_Performance"].mean()
    )
    fig = px.bar(
        academic,
        x="Placement_Label",
        y="Academic_Performance",
        color="Placement_Label",
        color_discrete_map={
            "Placed": "#7c8cff",
            "Not Placed": "#f59eae"
        },
        text_auto=".2f"
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e7edff",
        showlegend=False,
        xaxis_title="Placement Status",
        yaxis_title="Average Academic Performance"
    )
    fig.update_yaxes(gridcolor="#283550")
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("Internship Experience")
    internship = (
        filtered.groupby(["Placement_Label", "Internship_Experience"])
        .size()
        .reset_index(name="Students")
    )
    internship["Internship_Experience"] = (
        internship["Internship_Experience"].astype(str)
    )
    fig = px.bar(
        internship,
        x="Placement_Label",
        y="Students",
        color="Internship_Experience",
        barmode="group",
        color_discrete_sequence=["#7c8cff", "#42d6b5", "#f6bd60"]
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e7edff",
        xaxis_title="Placement Status",
        yaxis_title="Number of Students",
        legend_title="Internship Value"
    )
    fig.update_yaxes(gridcolor="#283550")
    st.plotly_chart(fig, use_container_width=True)

st.divider()
st.subheader("Student Dataset Preview")
st.caption("Preview of the filtered data. This does not modify your original CSV.")

display_columns = [
    "IQ", "Prev_Sem_Result", "CGPA", "Academic_Performance",
    "Internship_Experience", "Extra_Curricular_Score",
    "Communication_Skills", "Projects_Completed", "Placement_Label"
]
st.dataframe(
    filtered[display_columns].head(100),
    use_container_width=True,
    hide_index=True
)

csv_data = filtered[display_columns].to_csv(index=False).encode("utf-8")
st.download_button(
    label="⬇️ Download Filtered Dataset",
    data=csv_data,
    file_name="filtered_student_placement_data.csv",
    mime="text/csv"
)

st.markdown(
    '<p class="small-note">Student Placement Prediction System · Dashboard insights describe the supplied dataset and are not a guarantee of future placement outcomes.</p>',
    unsafe_allow_html=True
)
