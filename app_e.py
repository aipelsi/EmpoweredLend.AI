import streamlit as st
import pandas as pd
import numpy as np
import gdown
from tensorflow.keras.models import load_model
import joblib
import base64

# Define function to load resources
@st.cache_resource
def load_resources():
    model_url = 'https://drive.google.com/uc?id=1VPaz8JOudnGOwJw-IjhRYYSmk7SnHtDB'
    model_output = 'model.h5'
    gdown.download(model_url, model_output, quiet=False)
    model = load_model(model_output)

    scaler_url = 'https://drive.google.com/uc?id=1-n1VUFuwSPakfzx2SRogr5NhN4ZBzwW0'
    scaler_output = 'scaler.joblib'
    gdown.download(scaler_url, scaler_output, quiet=False)
    scaler = joblib.load(scaler_output)
    return model, scaler

model, scaler = load_resources()

# Define model columns
model_columns = ['GrossApproval', 'SBAGuaranteedApproval', 'InitialInterestRate', 'TermInMonths', 'JobsSupported', 'FixedOrVariableInterestInd_V', 'BusinessType_INDIVIDUAL', 'BusinessType_PARTNERSHIP']

def convert_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return f"data:image/png;base64,{encoded_string}"

image_base64 = convert_image_to_base64("photo.png")  # Adjust the path as necessary

# Custom CSS to style the app
st.markdown("""
    <style>
    body {
        background-color: black;
    }
    .reportview-container .main .block-container {
        background-color: white;
        color: black;
        padding: 2rem;
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# Displaying the app's title and logo
st.markdown(f"""
    <div style="text-align: center;">
        <img src="{image_base64}" alt="Logo" style="height: 80px;">
        <h1 style="color: black; text-align: center;">EmpowerLend.AI</h1>
        <p style="color: grey; font-style: italic;">Empowering Women-Owned Small Businesses</p>
    </div>
    """, unsafe_allow_html=True)

# Collecting user input
with st.form("loan_form"):
    st.write("## Personal Information")
    first_name = st.text_input("First Name")
    last_name = st.text_input("Last Name")
    age = st.number_input("Age", min_value=18, max_value=100, value=30, step=1)
    business_purpose = st.text_area("Business Purpose", height=100)

    st.write("## Loan Details")
    gross_approval = st.number_input('Amount Desired', min_value=0.0, max_value=100000.0, value=50000.0, format="%.2f")
    sba_guaranteed_approval = st.number_input('SBA Guaranteed Approval if Applicable', min_value=0.0, max_value=150000.0, value=25000.0, format="%.2f")
    initial_interest_rate = st.number_input('Initial Interest Rate Desired', min_value=0.0, max_value=20.0, value=5.0, format="%.2f")
    term_in_months = st.number_input('Term in Months Desired', min_value=0, max_value=120, value=120)
    jobs_supported = st.number_input('Jobs Supported', min_value=0, max_value=1000, value=1)
    fixed_or_variable_interest = st.selectbox('Interest Type', ['Variable', 'Fixed'])
    business_type_individual = st.radio('Is Individual Business?', ['Yes', 'No'])
    business_type_partnership = st.radio('Is Partnership?', ['Yes', 'No'])

    submitted = st.form_submit_button("Submit")
    if submitted:
        input_data = np.array([[gross_approval, sba_guaranteed_approval, initial_interest_rate, term_in_months, jobs_supported, 1 if fixed_or_variable_interest == 'Fixed' else 0, 1 if business_type_individual == 'Yes' else 0, 1 if business_type_partnership == 'Yes' else 0]])
        input_df = pd.DataFrame(input_data, columns=model_columns)
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)
        result = prediction[0][0]
        if result > 0.7:
            st.success('Congratulations, you are approved! A representative will contact you shortly to assist you with your loan request.')
        else:
            st.error('We cannot approve your request at the moment. But we will reach out to help you navigate other options.')





