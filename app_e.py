import streamlit as st
import pandas as pd
import numpy as np
import requests
from io import BytesIO
from tensorflow.keras.models import load_model
import joblib

# Function to download and load resources
@st.cache(allow_output_mutation=True)
def load_resources():
    # GitHub raw URL to the model
    model_url = 'https://github.com/aipelsi/EmpoweredLend.AI/raw/main/model.h5'
    response = requests.get(model_url)
    model = load_model(BytesIO(response.content))  # Assuming it's an H5 file

    # GitHub raw URL to the scaler
    scaler_url = 'https://github.com/aipelsi/EmpoweredLend.AI/raw/main/scaler.joblib'
    response = requests.get(scaler_url)
    scaler = joblib.load(BytesIO(response.content))

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
    
    age = st.number_input("Borrower's Age", min_value=18, max_value=100, value=30, step=1)
    city = st.text_input("City")
    for_profit = st.selectbox('For-Profit Status', ['Yes', 'No'])
    financial_education_needed = st.selectbox('Do you need financial education?', ['Yes', 'No'])
    gross_approval = st.number_input('Amount Requested', min_value=0, max_value=1000000, value=50000, step=1000)
    fixed_or_variable = st.selectbox('Fixed or Variable Interest', ['Fixed', 'Variable'])
    term_in_months = st.number_input('Term in Months', min_value=0, max_value=360, value=120, step=1)
    business_type = st.selectbox('Business Type', ['Type1', 'Type2', 'Type3'])
    jobs_supported = st.number_input('Jobs Supported', min_value=0, max_value=1000, value=10, step=1)

    submitted = st.form_submit_button("Predict")
    if submitted:
        input_data = {
            'GrossApproval': [gross_approval],
            'FixedOrVariableInterestInd': [1 if fixed_or_variable == 'Fixed' else 0],
            'TermInMonths': [term_in_months],
            'BusinessType_Type1': [1 if business_type == 'Type1' else 0],
            'BusinessType_Type2': [1 if business_type == 'Type2' else 0],
            'BusinessType_Type3': [1 if business_type == 'Type3' else 0],
            'JobsSupported': [jobs_supported]
        }
        processed_data = scaler.transform(pd.DataFrame(input_data))
        prediction = model.predict(processed_data)
        result = prediction[0][0]
        if result > 0.5:
            st.success('The loan is likely to be paid back.')
        else:
            st.error('There is a high risk the loan will not be paid back.')

            st.success('The loan is likely to be paid back.')
        else:
            st.error('There is a high risk the loan will not be paid back.')
