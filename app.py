import streamlit as st
from plotly.subplots import make_subplots
import plotly.graph_objects as go
st.set_page_config(
    page_title="Manufacturing Quality Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("Manufacturing Quality Analytics Dashboard")

st.write(
    "Interactive quality dashboard using synthetic manufacturing inspection data."
)

st.info(
    "Educational portfolio project — all manufacturing data is synthetic."
)
import pandas as pd

# Synthetic manufacturing inspection data
data = {
    "Defect Type": [
        "Bore Diameter",
        "Surface Finish",
        "Runout",
        "Burr",
        "Scratch"
    ],
    "Defect Count": [16, 5, 3, 2, 2]
}

df = pd.DataFrame(data)

# Production totals
total_inspected = 240
total_defects = df["Defect Count"].sum()
defect_rate = (total_defects / total_inspected) * 100
first_pass_yield = ((total_inspected - total_defects) / total_inspected) * 100

st.subheader("Quality Performance")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Units Inspected", total_inspected)
col2.metric("Nonconforming", total_defects)
col3.metric("Defect Rate", f"{defect_rate:.1f}%")
col4.metric("First Pass Yield", f"{first_pass_yield:.1f}%")
# Pareto Analysis


st.subheader("Defect Pareto Analysis")

# Sort defects from highest to lowest
pareto_df = df.sort_values(
    by="Defect Count",
    ascending=False
).copy()

# Calculate cumulative percentage
pareto_df["Cumulative %"] = (
    pareto_df["Defect Count"].cumsum()
    / pareto_df["Defect Count"].sum()
    * 100
)


# Create a true Pareto chart
fig = make_subplots(specs=[[{"secondary_y": True}]])

# Defect count bars
fig.add_trace(
    go.Bar(
        x=pareto_df["Defect Type"],
        y=pareto_df["Defect Count"],
        name="Defect Count",
        text=pareto_df["Defect Count"],
        textposition="outside"
    ),
    secondary_y=False
)

# Cumulative percentage line
fig.add_trace(
    go.Scatter(
        x=pareto_df["Defect Type"],
        y=pareto_df["Cumulative %"],
        name="Cumulative %",
        mode="lines+markers"
    ),
    secondary_y=True
)

# 80% reference line
fig.add_hline(
    y=80,
    line_dash="dash",
    annotation_text="80% Threshold",
    secondary_y=True
)

fig.update_layout(
    title="Pareto Analysis — Defects by Category",
    xaxis_title="Defect Type",
    legend_title="Metric"
)

fig.update_yaxes(
    title_text="Defect Count",
    secondary_y=False
)

fig.update_yaxes(
    title_text="Cumulative %",
    range=[0, 105],
    ticksuffix="%",
    secondary_y=True
)

st.plotly_chart(fig, use_container_width=True)
    
    
  




# Show the Pareto data
st.dataframe(
    pareto_df,
    hide_index=True,
    use_container_width=True
)
