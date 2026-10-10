import streamlit as st
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px

from quality_core import (
    DEMO_BORE_MEASUREMENTS,
    DEMO_DEFECT_COUNTS,
    DEMO_DEFECT_TYPES,
    DEMO_TOTAL_INSPECTED,
    RECOMMENDED_TOOL_LIFE,
    capability_summary,
    defect_rate as calc_defect_rate,
    first_pass_yield as calc_first_pass_yield,
    generate_synthetic_dataset,
    ml_decision,
    ml_quality_assessment,
    pareto_analysis,
    simulate_bore_diameter,
    spc_control_limits,
    tool_life_status,
)

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
st.set_page_config(
    page_title="Manufacturing Quality Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("Manufacturing Quality Analytics Dashboard")

with st.sidebar:
    st.header("Dashboard Navigation")

    with st.expander("Quality Analytics", expanded=True):
        st.markdown(
            """
            - [Quality Performance](#quality-performance)
            - [Defect Pareto Analysis](#defect-pareto-analysis)
            - [Process Capability](#process-capability)
            """
        )

    with st.expander("Root Cause & Improvement"):
        st.markdown(
            """
           - [Root Cause Analysis](#root-cause-analysis)
            - [Corrective / Preventive Action](#corrective-preventive-action)
           - [Tool-Life Risk Model](#tool-life-risk-model)
            """
        )

    with st.expander("Predictive Quality"):
        st.markdown(
            """
            - [Predictive Quality Impact](#predictive-quality-impact)
            - [Tool Wear vs. Bore Diameter](#tool-wear-bore-diameter)
            """
        )

    with st.expander("Machine Learning"):
        st.markdown(
            """
            - [Predictive Quality Model](#predictive-quality-model)
            - [Feature Importance](#feature-importance)
            - [Live ML Process Simulator](#live-ml-process-simulator)
            - [ML Decision Support](#ml-decision-support)
            - [ML Model Validation](#ml-model-validation)
            """
        )

    st.divider()

    st.caption(
        "Portfolio demonstration using synthetic manufacturing data."
    )

st.info(
    "Portfolio Project: All manufacturing and machine-learning data used in this "
    "dashboard are synthetic and intended for educational demonstration."
)


        
    



# Synthetic inspection data lives in quality_core so the demo numbers are
# unit-tested and shared with the test-suite (see DEMO_* constants).
df = pd.DataFrame({
    "Defect Type": DEMO_DEFECT_TYPES,
    "Defect Count": DEMO_DEFECT_COUNTS,
})

# Production totals
total_inspected = DEMO_TOTAL_INSPECTED
total_defects = df["Defect Count"].sum()
defect_rate = calc_defect_rate(total_defects, total_inspected)
first_pass_yield = calc_first_pass_yield(total_defects, total_inspected)
st.markdown(
    '<div id="quality-performance"></div>',
    unsafe_allow_html=True
)


st.subheader("Quality Performance")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Units Inspected", total_inspected)
col2.metric("Nonconforming", total_defects)
col3.metric("Defect Rate", f"{defect_rate:.1f}%")
col4.metric("First Pass Yield", f"{first_pass_yield:.1f}%")
# Pareto Analysis


st.markdown(
    '<div id="defect-pareto-analysis"></div>',
    unsafe_allow_html=True
)

st.subheader("Defect Pareto Analysis")


# Sort defects from highest to lowest and add the cumulative percentage.
# Delegated to quality_core.pareto_analysis (deterministic tie-breaking).
_pareto = pareto_analysis(
    df["Defect Type"].tolist(), df["Defect Count"].tolist()
)
pareto_df = pd.DataFrame([
    {
        "Defect Type": row["label"],
        "Defect Count": row["count"],
        "Cumulative %": row["cumulative_pct"],
    }
    for row in _pareto["rows"]
])


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

# Synthetic dimensional measurements in millimeters (shared constant)
measurements = DEMO_BORE_MEASUREMENTS

spc_df = pd.DataFrame({
    "Sample": range(1, len(measurements) + 1),
    "Measurement": measurements
})

# Calculate SPC statistics (3-sigma limits) and flag any out-of-control samples
_spc = spc_control_limits(measurements, sigma_multiplier=3.0)
process_mean = _spc["mean"]
process_std = _spc["std"]
ucl = _spc["ucl"]
lcl = _spc["lcl"]
out_of_control = _spc["out_of_control"]

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
# Out-of-control status (the chart draws limits; now violations are also flagged)
if out_of_control:
    st.error(
        f"Control-chart alert: {len(out_of_control)} measurement(s) sit "
        "outside the 3-sigma control limits — investigate the process."
    )
else:
    st.success(
        "Control-chart status: all measurements within the 3-sigma limits."
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

# Highlight any out-of-control samples (outside the 3-sigma limits)
if out_of_control:
    spc_fig.add_trace(
        go.Scatter(
            x=[spc_df["Sample"][i] for i in out_of_control],
            y=[spc_df["Measurement"][i] for i in out_of_control],
            mode="markers",
            name="Out of Control",
            marker=dict(color="red", size=12, symbol="x"),
        )
    )

spc_fig.update_layout(
    title="Synthetic Bore Diameter — 3-Sigma Demonstration",
    xaxis_title="Sample",
    yaxis_title="Diameter (mm)"
)

st.plotly_chart(spc_fig, use_container_width=True)
# Process Capability Analysis


st.markdown(
    '<div id="process-capability"></div>',
    unsafe_allow_html=True
)

st.subheader("Process Capability")


# Calculate capability indices.  capability_summary also derives the long-term
# performance indices (Pp/Ppk) and the target-aware Taguchi index (Cpm), which
# makes the "Target Diameter" sidebar input drive a real calculation.
_cap = capability_summary(measurements, lsl, usl, target=target)
cp = _cap["cp"]
cpk = _cap["cpk"]
pp = _cap["pp"]
ppk = _cap["ppk"]
cpm = _cap["cpm"]
cap_status = _cap["status"]

# Display capability metrics
cap_col1, cap_col2, cap_col3, cap_col4 = st.columns(4)

cap_col1.metric("LSL", f"{lsl:.2f} mm")
cap_col2.metric("USL", f"{usl:.2f} mm")
cap_col3.metric("Cp", f"{cp:.2f}")
cap_col4.metric("Cpk", f"{cpk:.2f}")

cap2_col1, cap2_col2, cap2_col3 = st.columns(3)
cap2_col1.metric("Pp (long-term)", f"{pp:.2f}")
cap2_col2.metric("Ppk (long-term)", f"{ppk:.2f}")
cap2_col3.metric("Cpm (target-aware)", f"{cpm:.2f}")

st.caption(
    "Cp/Cpk use the short-term (sample) sigma; Pp/Ppk use the long-term "
    "(overall) sigma.  Cpm additionally penalises drift away from the "
    "Target Diameter."
)

# Capability status
st.subheader("Capability Assessment")

if cap_status == "Capable":
    st.success("Demonstration Status: Capable")
elif cap_status == "Marginal":
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
st.markdown(
    '<div id="corrective-preventive-action"></div>',
    unsafe_allow_html=True
)
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
st.markdown(
    '<div id="root-cause-analysis"></div>',
    unsafe_allow_html=True
)
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
st.markdown(
    '<div id="tool-life-risk-model"></div>',
    unsafe_allow_html=True
)
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

# Synthetic recommended tool-life limit (shared with quality_core)
recommended_tool_life = RECOMMENDED_TOOL_LIFE

tool_life_used = (tool_cycles / recommended_tool_life) * 100

st.metric(
    "Tool Life Used",
    f"{tool_life_used:.1f}%"
)

# Tool-life risk classification (delegated to quality_core.tool_life_status)
_tl_status = tool_life_status(tool_cycles)
if _tl_status["level"] == "ok":
    st.success("Tool Status: " + _tl_status["label"])
elif _tl_status["level"] == "warn":
    st.warning("Tool Status: " + _tl_status["label"])
else:
    st.error("Tool Status: " + _tl_status["label"])
# Predictive Quality Impact Model
st.markdown(
    '<div id="predictive-quality-impact"></div>',
    unsafe_allow_html=True
)
st.subheader("Predictive Quality Impact")

st.write(
    "Simulate how increasing cutting-tool usage may influence "
    "bore diameter stability and dimensional defect risk."
)

# Synthetic bore-diameter model.  The piecewise drift function lives in
# quality_core so the slider below and the chart further down cannot drift apart.
nominal_bore = 25.00
upper_spec = 25.10
lower_spec = 24.90

# Tool wear begins influencing dimensional stability
predicted_bore = simulate_bore_diameter(tool_cycles)

st.metric(
    "Predicted Bore Diameter",
    f"{predicted_bore:.3f} mm"
)

if lower_spec <= predicted_bore <= upper_spec:
    st.success("Dimensional Status: Within Specification")
else:
    st.error("Dimensional Status: Out of Specification")
# Tool Wear vs. Bore Diameter Visualization
st.markdown(
    '<div id="tool-wear-bore-diameter"></div>',
    unsafe_allow_html=True
)
st.subheader("Tool Wear vs. Bore Diameter")

st.write(
    "Visualize how predicted bore diameter changes as the cutting tool "
    "progresses through its operating life."
)

# Generate synthetic tool-life curve
cycle_range = list(range(0, 1001, 25))

# Same shared model function as the slider (single source of truth for drift).
predicted_bores = [simulate_bore_diameter(cycle) for cycle in cycle_range]

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
# Machine Learning Predictive Quality Model
st.markdown(
    '<div id="predictive-quality-model"></div>',
    unsafe_allow_html=True
)
st.subheader("Machine Learning — Predictive Quality Model")

st.write(
    "Train a machine-learning model on synthetic CNC process data to predict "
    "bore diameter from operating conditions."
)

# Reproducible synthetic manufacturing dataset, generated in quality_core.
# Same numbers every render, and no longer mutates NumPy's global RNG seed.
ml_data = pd.DataFrame(generate_synthetic_dataset(sample_size=500, seed=42))

# ML features and prediction target
X = ml_data[
    [
        "Tool Cycles",
        "Spindle Speed",
        "Feed Rate",
        "Vibration",
        "Temperature"
    ]
]

y = ml_data["Bore Diameter"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# Train Random Forest regression model
ml_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

ml_model.fit(X_train, y_train)

# Evaluate model
y_pred = ml_model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

col1, col2, col3 = st.columns(3)

col1.metric(
    "Training Samples",
    len(X_train)
)

col2.metric(
    "Mean Absolute Error",
    f"{mae:.4f} mm"
)

col3.metric(
    "R² Score",
    f"{r2:.3f}"
)
# Machine Learning Feature Importance
st.markdown(
    '<div id="feature-importance"></div>',
    unsafe_allow_html=True
)
st.subheader("ML Feature Importance")

st.write(
    "See which manufacturing process variables have the greatest influence "
    "on the machine-learning model's bore-diameter predictions."
)

feature_importance = pd.DataFrame({
    "Process Variable": X.columns,
    "Importance": ml_model.feature_importances_
}).sort_values(
    by="Importance",
    ascending=True
)

fig_importance = px.bar(
    feature_importance,
    x="Importance",
    y="Process Variable",
    orientation="h",
    title="Random Forest Feature Importance"
)

fig_importance.update_layout(
    xaxis_title="Relative Importance",
    yaxis_title="Process Variable"
)

st.plotly_chart(
    fig_importance,
    use_container_width=True
)

st.caption(
    "Feature importance reflects this synthetic training dataset and "
    "should not be interpreted as validated real-world causal relationships."
)
# Live ML Process Simulator
st.markdown(
    '<div id="live-ml-process-simulator"></div>',
    unsafe_allow_html=True
)
st.subheader("Live ML Process Simulator")

st.write(
    "Adjust CNC operating conditions and let the trained Random Forest model "
    "predict the resulting bore diameter in real time."
)

# Interactive process controls
sim_col1, sim_col2 = st.columns(2)

with sim_col1:
    sim_tool_cycles = st.slider(
        "ML Tool Cycles",
        min_value=0,
        max_value=1000,
        value=400,
        step=25
    )

    sim_spindle_speed = st.slider(
        "Spindle Speed (RPM)",
        min_value=1800,
        max_value=3200,
        value=2500,
        step=50
    )

    sim_feed_rate = st.slider(
        "Feed Rate",
        min_value=80.0,
        max_value=180.0,
        value=130.0,
        step=5.0
    )

with sim_col2:
    sim_vibration = st.slider(
        "Vibration",
        min_value=0.5,
        max_value=4.0,
        value=2.0,
        step=0.1
    )

    sim_temperature = st.slider(
        "Process Temperature (°C)",
        min_value=20.0,
        max_value=45.0,
        value=30.0,
        step=1.0
    )

# Build input using the same feature names used during training
live_input = pd.DataFrame({
    "Tool Cycles": [sim_tool_cycles],
    "Spindle Speed": [sim_spindle_speed],
    "Feed Rate": [sim_feed_rate],
    "Vibration": [sim_vibration],
    "Temperature": [sim_temperature]
})

# Generate live ML prediction
live_prediction = ml_model.predict(live_input)[0]

st.metric(
    "ML Predicted Bore Diameter",
    f"{live_prediction:.3f} mm"
)

# Compare ML prediction with engineering specification limits
_assessment = ml_quality_assessment(live_prediction, lower_spec, upper_spec)
if _assessment == "out_high":
    st.error(
        "ML Quality Risk: Predicted bore diameter exceeds the "
        "25.100 mm upper specification limit."
    )
elif _assessment == "out_low":
    st.error(
        "ML Quality Risk: Predicted bore diameter is below the "
        "24.900 mm lower specification limit."
    )
elif _assessment == "near_upper":
    st.warning(
        "ML Quality Risk: Bore diameter is within specification "
        "but approaching the upper specification limit."
    )
else:
    st.success(
        "ML Quality Status: Predicted bore diameter is within specification."
    )

st.caption(
    "Educational ML simulation only. Predictions are generated from a "
    "Random Forest trained on synthetic manufacturing data."
)
# ML Decision Support
st.markdown(
    '<div id="ml-decision-support"></div>',
    unsafe_allow_html=True
)
st.subheader("ML Decision Support")

st.write(
    "Combine preventive tool-life limits with the machine-learning prediction "
    "to generate an actionable quality recommendation."
)

# Calculate preventive tool-life status
sim_tool_life_percent = (
    sim_tool_cycles / recommended_tool_life
) * 100

decision_col1, decision_col2 = st.columns(2)

decision_col1.metric(
    "Tool Life Used",
    f"{sim_tool_life_percent:.1f}%"
)

decision_col2.metric(
    "ML Predicted Bore",
    f"{live_prediction:.3f} mm"
)

# Decision-support logic (single source of truth in quality_core.ml_decision)
_action = ml_decision(sim_tool_cycles, live_prediction, lower_spec, upper_spec)
if _action == "stop_and_inspect":
    st.error(
        "ACTION: Stop and Inspect — ML predicts a dimensional "
        "condition outside the engineering specification limits."
    )
elif _action == "replace_tool":
    st.error(
        "ACTION: Replace Tool — Preventive tool-life limit has been reached "
        "and continued production increases dimensional risk."
    )
elif _action == "prepare_tool_change":
    st.warning(
        "ACTION: Prepare Tool Change — Product remains within specification, "
        "but the ML model indicates increasing dimensional risk."
    )
elif _action == "increase_monitoring":
    st.warning(
        "ACTION: Increase Monitoring — Tool is approaching its preventive "
        "replacement interval."
    )
else:
    st.success(
        "ACTION: Continue Production — Tool life and ML-predicted dimensional "
        "performance remain within the current operating criteria."
    )

st.caption(
    "Decision support is an educational demonstration based on synthetic "
    "manufacturing data and predefined engineering thresholds."
)
# Actual vs. Predicted Model Validation
st.markdown(
    '<div id="ml-model-validation"></div>',
    unsafe_allow_html=True
)
st.subheader("ML Model Validation")

st.write(
    "Compare actual synthetic test measurements with Random Forest predictions "
    "to evaluate how closely the model reproduces unseen bore-diameter results."
)

validation_df = pd.DataFrame({
    "Actual Bore Diameter": y_test.values,
    "Predicted Bore Diameter": y_pred
})

fig_validation = px.scatter(
    validation_df,
    x="Actual Bore Diameter",
    y="Predicted Bore Diameter",
    title="Actual vs. Predicted Bore Diameter"
)

# Perfect-prediction reference line
validation_min = min(
    validation_df["Actual Bore Diameter"].min(),
    validation_df["Predicted Bore Diameter"].min()
)

validation_max = max(
    validation_df["Actual Bore Diameter"].max(),
    validation_df["Predicted Bore Diameter"].max()
)

fig_validation.add_shape(
    type="line",
    x0=validation_min,
    y0=validation_min,
    x1=validation_max,
    y1=validation_max,
    line=dict(
        dash="dash"
    )
)

fig_validation.update_layout(
    xaxis_title="Actual Bore Diameter (mm)",
    yaxis_title="Predicted Bore Diameter (mm)"
)

st.plotly_chart(
    fig_validation,
    use_container_width=True
)

st.caption(
    "Points closer to the diagonal reference line indicate closer agreement "
    "between synthetic test measurements and ML predictions."
)
