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

# 80% Pareto threshold on the percentage axis
fig.add_trace(
    go.Scatter(
        x=pareto_df["Defect Type"],
        y=[80] * len(pareto_df),
        name="80% Threshold",
        mode="lines",
        line=dict(dash="dash")
    ),
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
# Statistical Process Control
st.subheader("Statistical Process Control")

st.write(
    "Synthetic bore diameter measurements used to demonstrate process variation and control limits."
)

# Synthetic dimensional measurements in millimeters
measurements = [
    25.01, 24.98, 25.03, 25.00, 24.97,
    25.02, 25.04, 24.99, 25.01, 24.96,
    25.00, 25.03, 24.98, 25.02, 25.01,
    24.99, 25.04, 25.00, 24.97, 25.02
]

spc_df = pd.DataFrame({
    "Sample": range(1, len(measurements) + 1),
    "Measurement": measurements
})

# Calculate SPC statistics
process_mean = spc_df["Measurement"].mean()
process_std = spc_df["Measurement"].std()

ucl = process_mean + (3 * process_std)
lcl = process_mean - (3 * process_std)

# Interactive engineering specification limits
st.sidebar.header("Process Settings")

lsl = st.sidebar.number_input(
    "Lower Specification Limit (mm)",
    value=24.90,
    step=0.01,
    format="%.2f"
)

target = st.sidebar.number_input(
    "Target Diameter (mm)",
    value=25.00,
    step=0.01,
    format="%.2f"
)

usl = st.sidebar.number_input(
    "Upper Specification Limit (mm)",
    value=25.10,
    step=0.01,
    format="%.2f"
)
# Display SPC metrics
spc_col1, spc_col2, spc_col3, spc_col4 = st.columns(4)

spc_col1.metric("Process Mean", f"{process_mean:.3f} mm")
spc_col2.metric("Std. Deviation", f"{process_std:.3f} mm")
spc_col3.metric("UCL", f"{ucl:.3f} mm")
spc_col4.metric("LCL", f"{lcl:.3f} mm")
st.dataframe(
    spc_df,
    hide_index=True,
    use_container_width=True
)
# SPC visualization
st.subheader("Bore Diameter Control Chart")

spc_fig = go.Figure()

# Measurement data
spc_fig.add_trace(
    go.Scatter(
        x=spc_df["Sample"],
        y=spc_df["Measurement"],
        mode="lines+markers",
        name="Measurement"
    )
)

# Process mean
spc_fig.add_trace(
    go.Scatter(
        x=spc_df["Sample"],
        y=[process_mean] * len(spc_df),
        mode="lines",
        name="Process Mean"
    )
)

# Upper control limit
spc_fig.add_trace(
    go.Scatter(
        x=spc_df["Sample"],
        y=[ucl] * len(spc_df),
        mode="lines",
        name="UCL",
        line=dict(dash="dash")
    )
)

# Lower control limit
spc_fig.add_trace(
    go.Scatter(
        x=spc_df["Sample"],
        y=[lcl] * len(spc_df),
        mode="lines",
        name="LCL",
        line=dict(dash="dash")
    )
)

spc_fig.update_layout(
    title="Synthetic Bore Diameter — 3-Sigma Demonstration",
    xaxis_title="Sample",
    yaxis_title="Diameter (mm)"
)

st.plotly_chart(spc_fig, use_container_width=True)
# Process Capability Analysis
st.subheader("Process Capability")

# Calculate Cp and Cpk
cp = (usl - lsl) / (6 * process_std)

cpu = (usl - process_mean) / (3 * process_std)
cpl = (process_mean - lsl) / (3 * process_std)

cpk = min(cpu, cpl)

# Display capability metrics
cap_col1, cap_col2, cap_col3, cap_col4 = st.columns(4)

cap_col1.metric("LSL", f"{lsl:.2f} mm")
cap_col2.metric("USL", f"{usl:.2f} mm")
cap_col3.metric("Cp", f"{cp:.2f}")
cap_col4.metric("Cpk", f"{cpk:.2f}")
