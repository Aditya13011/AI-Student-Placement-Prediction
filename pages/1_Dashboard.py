
import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="Placement Analytics Dashboard",
    page_icon="📊",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "college_student_placement_dataset.csv"

st.title("📊 Placement Analytics Dashboard")
st.caption("AI-Based Student Placement Prediction System")
st.divider()

# --------------------------------------------------
# LOAD DATASET
# --------------------------------------------------
@st.cache_data
def load_data(path):
    data = pd.read_csv(path)

    required = [
        "CGPA",
        "IQ",
        "Communication_Skills",
        "Projects_Completed",
        "Internship_Experience",
        "Placement"
    ]

    missing = [col for col in required if col not in data.columns]

    if missing:
        raise ValueError(
            "Dataset is missing required columns: "
            + ", ".join(missing)
        )

    return data

try:
    df = load_data(DATA_PATH)
except Exception as error:
    st.error(f"Unable to load the dataset: {error}")
    st.stop()

# --------------------------------------------------
# DATA PREPARATION
# --------------------------------------------------
df["Placement"] = (
    df["Placement"]
    .astype(str)
    .str.strip()
    .str.title()
)

df = df[df["Placement"].isin(["Yes", "No"])].copy()

if df.empty:
    st.error("No valid Yes/No placement records were found.")
    st.stop()

# --------------------------------------------------
# SIDEBAR FILTERS
# --------------------------------------------------
st.sidebar.header("🔎 Dashboard Filters")

min_cgpa = float(df["CGPA"].min())
max_cgpa = float(df["CGPA"].max())

cgpa_range = st.sidebar.slider(
    "CGPA range",
    min_value=min_cgpa,
    max_value=max_cgpa,
    value=(min_cgpa, max_cgpa)
)

internship_options = sorted(
    df["Internship_Experience"]
    .dropna()
    .astype(str)
    .unique()
    .tolist()
)

selected_internships = st.sidebar.multiselect(
    "Internship experience",
    options=internship_options,
    default=internship_options
)

filtered = df[
    df["CGPA"].between(cgpa_range[0], cgpa_range[1])
].copy()

filtered["Internship_Experience"] = (
    filtered["Internship_Experience"].astype(str)
)

if selected_internships:
    filtered = filtered[
        filtered["Internship_Experience"].isin(
            selected_internships
        )
    ]
else:
    filtered = filtered.iloc[0:0]

if filtered.empty:
    st.warning("No students match the selected filters.")
    st.stop()

# --------------------------------------------------
# KEY PERFORMANCE INDICATORS
# --------------------------------------------------
total_students = len(filtered)
placed_students = int((filtered["Placement"] == "Yes").sum())
not_placed_students = int((filtered["Placement"] == "No").sum())
placement_rate = placed_students / total_students * 100

col1, col2, col3, col4 = st.columns(4)

col1.metric("👨‍🎓 Students", f"{total_students:,}")
col2.metric("✅ Placed", f"{placed_students:,}")
col3.metric("❌ Not Placed", f"{not_placed_students:,}")
col4.metric("📈 Placement Rate", f"{placement_rate:.2f}%")

st.divider()

# --------------------------------------------------
# PLACEMENT DISTRIBUTION
# --------------------------------------------------
left, right = st.columns(2)

with left:
    st.subheader("Placement Distribution")

    placement_counts = (
        filtered["Placement"]
        .value_counts()
        .rename_axis("Placement Status")
        .reset_index(name="Students")
    )

    fig = px.pie(
        placement_counts,
        names="Placement Status",
        values="Students",
        hole=0.55,
        title="Placed vs Not Placed"
    )

    fig.update_traces(
        textposition="inside",
        textinfo="percent+label"
    )

    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("CGPA vs Placement")

    fig = px.box(
        filtered,
        x="Placement",
        y="CGPA",
        color="Placement",
        points="outliers",
        title="CGPA Distribution by Placement Status"
    )

    st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------
# SKILL ANALYSIS
# --------------------------------------------------
st.subheader("Student Skills Analysis")

skill_column = st.selectbox(
    "Choose a feature",
    [
        "IQ",
        "Communication_Skills",
        "Projects_Completed"
    ]
)

fig = px.histogram(
    filtered,
    x=skill_column,
    color="Placement",
    barmode="overlay",
    opacity=0.75,
    title=f"{skill_column.replace('_', ' ')} Distribution",
    marginal="box"
)

st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------
# INTERNSHIP ANALYSIS
# --------------------------------------------------
st.subheader("Internship Experience Analysis")

internship_summary = (
    filtered.groupby(
        "Internship_Experience",
        dropna=False
    )["Placement"]
    .apply(
        lambda values: (
            values.eq("Yes").mean() * 100
        )
    )
    .reset_index(name="Observed Placement Rate (%)")
)

fig = px.bar(
    internship_summary,
    x="Internship_Experience",
    y="Observed Placement Rate (%)",
    text_auto=".2f",
    title="Observed Placement Rate by Internship Experience"
)

st.plotly_chart(fig, use_container_width=True)

st.caption(
    "These charts describe the uploaded dataset. "
    "Observed placement rates are historical dataset statistics, "
    "not guarantees of future placement."
)

# --------------------------------------------------
# DATA TABLE AND DOWNLOAD
# --------------------------------------------------
with st.expander("📄 View filtered student records"):
    st.dataframe(
        filtered,
        use_container_width=True,
        hide_index=True
    )

    csv_data = filtered.to_csv(index=False).encode("utf-8")

    st.download_button(
        "⬇️ Download filtered data as CSV",
        data=csv_data,
        file_name="filtered_student_placement_data.csv",
        mime="text/csv"
    )

st.divider()
st.caption("Student Placement Intelligence System | Analytics")
