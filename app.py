import streamlit as st

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
