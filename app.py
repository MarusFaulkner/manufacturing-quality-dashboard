import streamlit as st
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px

import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
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
st.sidebar.header("What-If Specification Analysis")
st.sidebar.caption(
    "Adjust the synthetic specification limits to explore how tolerance changes affect process capability."
)
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
# Capability status
st.subheader("Capability Assessment")

if cpk >= 1.33:
    st.success("Demonstration Status: Capable")
elif cpk >= 1.00:
    st.warning("Demonstration Status: Marginal")
else:
    st.error("Demonstration Status: Not Capable")
# Defect Trend Analysis
st.subheader("Defect Trend Analysis")

st.write(
    "Synthetic weekly defect data used to demonstrate quality performance over time."
)

# Synthetic weekly production data
trend_data = {
    "Week": ["Week 1", "Week 2", "Week 3", "Week 4", "Week 5", "Week 6"],
    "Units Inspected": [210, 225, 230, 240, 250, 260],
    "Defects": [31, 29, 27, 28, 23, 20]
}

trend_df = pd.DataFrame(trend_data)

# Calculate weekly defect rate
trend_df["Defect Rate %"] = (
    trend_df["Defects"] / trend_df["Units Inspected"]
) * 100
# Create defect rate trend chart
trend_fig = go.Figure()

trend_fig.add_trace(
    go.Scatter(
        x=trend_df["Week"],
        y=trend_df["Defect Rate %"],
        mode="lines+markers",
        name="Defect Rate"
    )
)

trend_fig.update_layout(
    title="Weekly Defect Rate Trend",
    xaxis_title="Production Week",
    yaxis_title="Defect Rate (%)"
)

st.plotly_chart(trend_fig, use_container_width=True)
# CAPA Tracking
st.subheader("Corrective and Preventive Action (CAPA)")

st.write(
    "Synthetic CAPA records used to demonstrate corrective action tracking and effectiveness verification."
)

capa_data = {
    "CAPA ID": ["CAPA-001", "CAPA-002", "CAPA-003"],
    "Issue": [
        "Bore Diameter Variation",
        "Surface Finish Defect",
        "Excessive Runout"
    ],
    "Root Cause": [
        "Tool wear exceeded replacement interval",
        "Coolant concentration variation",
        "Fixture alignment variation"
    ],
    "Corrective Action": [
        "Reduce tool replacement interval",
        "Standardize coolant concentration checks",
        "Add fixture alignment verification"
    ],
    "Status": [
        "Verified",
        "In Progress",
        "Open"
    ]
}

capa_df = pd.DataFrame(capa_data)
# Display CAPA tracking table
st.dataframe(
    capa_df,
    hide_index=True,
    use_container_width=True
)
# CAPA Effectiveness Verification
st.subheader("CAPA Effectiveness Verification")

before_defects = 16
after_defects = 4

defect_reduction = (
    (before_defects - after_defects) / before_defects
) * 100
verify_col1, verify_col2, verify_col3 = st.columns(3)

verify_col1.metric("Defects Before CAPA", before_defects)
verify_col2.metric("Defects After CAPA", after_defects)
verify_col3.metric("Defect Reduction", f"{defect_reduction:.1f}%")
# Effectiveness decision
if defect_reduction >= 50:
    st.success("CAPA Effectiveness: Effective")
else:
    st.warning("CAPA Effectiveness: Further Action Required")
# Root Cause Analysis
st.subheader("5 Whys Root Cause Analysis")

st.write(
    "Structured root cause investigation of the synthetic bore diameter variation."
)

why_data = {
    "Step": ["Why 1", "Why 2", "Why 3", "Why 4", "Why 5"],
    "Finding": [
        "Why did the bore diameter vary? Cutting dimensions drifted during production.",
        "Why did dimensions drift? Tool wear increased during the production run.",
        "Why did tool wear increase? The cutting tool remained in service too long.",
        "Why did the tool remain in service too long? The replacement interval was inadequate.",
        "Why was the interval inadequate? Tool-life data was not being used to optimize the replacement schedule."
    ]
}

why_df = pd.DataFrame(why_data)

st.dataframe(
    why_df,
    hide_index=True,
    use_container_width=True
)

st.subheader("Root Cause Conclusion")

st.error(
    "Root Cause Identified: Tool replacement intervals were not optimized using tool-life data."
)

st.subheader("Root Cause to CAPA Link")

st.info(
    "CAPA-001: Reduce the cutting-tool replacement interval and use tool-life data "
    "to establish a preventive replacement schedule."
)

# Root Cause Classification
st.subheader("Root Cause Classification")

rca_col1, rca_col2, rca_col3 = st.columns(3)

rca_col1.metric("6M Category", "Machine / Method")
rca_col2.metric("Failure Mode", "Cutting Tool Wear")
rca_col3.metric("System Cause", "Tool-Life Management")

st.write(
    "**Preventive Control:** Establish a data-driven tool replacement interval "
    "before dimensional drift produces nonconforming material."
)

st.write(
    "**Verification Method:** Monitor bore diameter measurements, defect rate, "
    "process capability, and recurrence after corrective action."
)
# Interactive Tool-Life Risk Model
st.subheader("Interactive Tool-Life Risk Model")

st.write(
    "Explore how increasing cutting-tool usage can influence synthetic "
    "tool-wear risk and dimensional stability."
)

tool_cycles = st.slider(
    "Cutting Tool Cycles",
    min_value=0,
    max_value=1000,
    value=400,
    step=25
)

# Synthetic recommended tool-life limit
recommended_tool_life = 800

tool_life_used = (tool_cycles / recommended_tool_life) * 100

st.metric(
    "Tool Life Used",
    f"{tool_life_used:.1f}%"
)

# Tool-life risk classification
if tool_life_used < 75:
    st.success("Tool Status: Normal — Continue Monitoring")
elif tool_life_used < 100:
    st.warning("Tool Status: Monitor — Approaching Preventive Replacement")
else:
    st.error("Tool Status: Replacement Recommended — Tool-Life Limit Reached")
# Predictive Quality Impact Model
st.subheader("Predictive Quality Impact")

st.write(
    "Simulate how increasing cutting-tool usage may influence "
    "bore diameter stability and dimensional defect risk."
)

# Synthetic bore-diameter model
nominal_bore = 25.00
upper_spec = 25.10
lower_spec = 24.90

# Tool wear begins influencing dimensional stability
if tool_cycles < 600:
    predicted_bore = nominal_bore
elif tool_cycles < 800:
    predicted_bore = nominal_bore + ((tool_cycles - 600) / 200) * 0.08
else:
    predicted_bore = 25.08 + ((tool_cycles - 800) / 200) * 0.07

st.metric(
    "Predicted Bore Diameter",
    f"{predicted_bore:.3f} mm"
)

if lower_spec <= predicted_bore <= upper_spec:
    st.success("Dimensional Status: Within Specification")
else:
    st.error("Dimensional Status: Out of Specification")
# Tool Wear vs. Bore Diameter Visualization
st.subheader("Tool Wear vs. Bore Diameter")

st.write(
    "Visualize how predicted bore diameter changes as the cutting tool "
    "progresses through its operating life."
)

# Generate synthetic tool-life curve
cycle_range = list(range(0, 1001, 25))
predicted_bores = []

for cycle in cycle_range:
    if cycle < 600:
        bore = nominal_bore
    elif cycle < 800:
        bore = nominal_bore + ((cycle - 600) / 200) * 0.08
    else:
        bore = 25.08 + ((cycle - 800) / 200) * 0.07

    predicted_bores.append(bore)

tool_wear_df = pd.DataFrame({
    "Tool Cycles": cycle_range,
    "Predicted Bore Diameter": predicted_bores
})

fig_tool_wear = px.line(
    tool_wear_df,
    x="Tool Cycles",
    y="Predicted Bore Diameter",
    markers=True,
    title="Predicted Bore Diameter vs. Tool Usage"
)

# Upper specification limit
fig_tool_wear.add_hline(
    y=upper_spec,
    line_dash="dash",
    annotation_text="USL 25.100 mm",
    annotation_position="top left"
)

# Lower specification limit
fig_tool_wear.add_hline(
    y=lower_spec,
    line_dash="dash",
    annotation_text="LSL 24.900 mm",
    annotation_position="bottom left"
)

# Preventive tool replacement point
fig_tool_wear.add_vline(
    x=recommended_tool_life,
    line_dash="dash",
    annotation_text="Preventive Replacement — 800 Cycles",
    annotation_position="top left"
)

# Current slider position
fig_tool_wear.add_scatter(
    x=[tool_cycles],
    y=[predicted_bore],
    mode="markers",
    marker=dict(size=14),
    name="Current Tool Position"
)

fig_tool_wear.update_layout(
    xaxis_title="Cutting Tool Cycles",
    yaxis_title="Bore Diameter (mm)",
    hovermode="x unified"
)

st.plotly_chart(
    fig_tool_wear,
    use_container_width=True
)
