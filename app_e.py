# -*- coding: utf-8 -*-
"""app_E.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1ApZQzt9CqwLy1YYO8GInEgpDvuivrlkP
"""

import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib
import gdown

# Function to download and load resources
@st.cache(allow_output_mutation=True)
def load_resources():
    # Download and load the model
    model_path = 'model.h5'
    gdown.download(model_path, quiet=False)
    model = load_model(model_path)

    # Load the scaler (assuming it's locally available or adjust if it's online)
    scaler_path = 'scaler.joblib'
    scaler = joblib.load(scaler_path)

    return model, scaler

model, scaler = load_resources()

# Custom CSS to style the app
st.markdown("""
    <style>
    .reportview-container {
        background: url("https://source.unsplash.com/weekly?water");  # Background image
        background-size: cover;
    }
    .big-font {
        font-size:28px !important;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown('<p class="big-font">Loan Repayment Prediction App</p>', unsafe_allow_html=True)

# Input Section
with st.form("my_form"):
    st.write("## Enter Loan Details")

    # Collect new input fields
    age = st.number_input("Borrower's Age", min_value=18, max_value=100, value=30, step=1)
    city = st.text_input("City")
    for_profit = st.selectbox('For-Profit Status', ['Yes', 'No'])
    financial_education_needed = st.selectbox('Do you need financial education?', ['Yes', 'No'])

    # Collect existing fields modified to match dataset
    gross_approval = st.number_input('Amount Requested', min_value=0, max_value=1000000, value=50000, step=1000)
    fixed_or_variable = st.selectbox('Fixed or Variable Interest', ['Fixed', 'Variable'])
    term_in_months = st.number_input('Term in Months', min_value=0, max_value=360, value=120, step=1)
    business_type = st.selectbox('Business Type', ['Type1', 'Type2', 'Type3'])  # Adjust with actual types
    jobs_supported = st.number_input('Jobs Supported', min_value=0, max_value=1000, value=10, step=1)

    submitted = st.form_submit_button("Predict")
    if submitted:
        # Prepare input data for prediction
        input_data = {
            'GrossApproval': [gross_approval],
            'FixedOrVariableInterestInd': [1 if fixed_or_variable == 'Fixed' else 0],  # Assuming binary encoding
            'TermInMonths': [term_in_months],
            'BusinessType_Type1': [1 if business_type == 'Type1' else 0],  # One-hot encoding manually
            'BusinessType_Type2': [1 if business_type == 'Type2' else 0],
            'BusinessType_Type3': [1 if business_type == 'Type3' else 0],
            'JobsSupported': [jobs_supported]
        }

        processed_data = scaler.transform(pd.DataFrame(input_data))

        # Make prediction
        prediction = model.predict(processed_data)
        result = prediction[0][0]

        # Display the result
        if result > 0.5:
            st.success('The loan is likely to be paid back.')
        else:
            st.error('There is a high risk the loan will not be paid back.')