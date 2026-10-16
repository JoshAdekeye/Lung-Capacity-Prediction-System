# Lung Capacity Prediction System
# Streamlit Application

import streamlit as st
import pandas as pd
from joblib import load


# Page Configuration
st.set_page_config(
    page_title="Lung Capacity Prediction System",
    page_icon="🫁",
    layout="centered"
)


# Load Trained Model
model = load("Best_Regression_Model.joblib")


# Title
st.title("🫁 Lung Capacity Prediction System")

st.write(
"""
This application predicts **Lung Capacity** using the
best machine learning regression model.
"""
)

st.divider()

# Sidebar Inputs
st.sidebar.header("Patient Information")

age = st.sidebar.number_input(
    "Age (Years)",
    min_value=3,
    max_value=25,
    value=12
)

height = st.sidebar.number_input(
    "Height (Inches)",
    min_value=40.0,
    max_value=80.0,
    value=60.0,
    step=0.1
)

gender = st.sidebar.selectbox(
    "Gender",
    ["Female", "Male"]
)

smoke = st.sidebar.selectbox(
    "Smoking Status",
    ["No", "Yes"]
)

caesarean = st.sidebar.selectbox(
    "Caesarean Birth",
    ["No", "Yes"]
)

# Encode Inputs
gender_value = 1 if gender == "Male" else 0
smoke_value = 1 if smoke == "Yes" else 0
caesarean_value = 1 if caesarean == "Yes" else 0

# Create Input DataFrame
input_data = pd.DataFrame({
    "Age": [age],
    "Height": [height],
    "Smoke": [smoke_value],
    "Gender": [gender_value],
    "Caesarean": [caesarean_value]
})

# Display Input
st.subheader("Entered Information")

st.dataframe(input_data, use_container_width=True)

# Prediction
if st.button("🔍 Predict Lung Capacity", use_container_width=True):

    prediction = model.predict(input_data)[0]

    st.success(
        f"### Predicted Lung Capacity: {prediction:.3f}"
    )

    st.subheader("Patient Summary")

    st.write(f"**Age:** {age} years")
    st.write(f"**Height:** {height} inches")
    st.write(f"**Gender:** {gender}")
    st.write(f"**Smoking:** {smoke}")
    st.write(f"**Caesarean Birth:** {caesarean}")

# Footer
st.divider()

st.caption(
    "Developed using Streamlit and Scikit-learn"
)