import streamlit as st
import pandas as pd
import gdown
from tensorflow.keras.models import load_model
import joblib
import numpy as np

# Function to download and load resources
@st.experimental_singleton
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

# Assuming model_columns are defined or loaded here
model_columns = ['GrossApproval', 'SBAGuaranteedApproval', 'InitialInterestRate',
                 'TermInMonths', 'JobsSupported', 'FixedOrVariableInterestInd_V',
                 'BusinessType_INDIVIDUAL', 'BusinessType_PARTNERSHIP']

# Direct link to your logo image
image_url = "https://drive.google.com/uc?export=view&id=1zOv_DjoL8a9aCFuZI32XlNSdMZd57ykF"

st.markdown(f"""
    <div style="text-align: center;">
        <img src="{image_url}" alt="Logo" style="height: 100px;">
        <h1 style="color: black; text-align: center;">EmpowerLend.AI</h1>
        <p style="color: grey; font-style: italic;">Empowering Women-Owned Small Businesses</p>
    </div>
    """, unsafe_allow_html=True)

with st.form("loan_form"):
    st.write("## Personal Information")
    first_name = st.text_input("First Name")
    last_name = st.text_input("Last Name")
    age = st.number_input("Age", min_value=18, max_value=100, value=30, step=1)
    business_purpose = st.text_area("Business Purpose", height=100)

    st.write("## Loan Details")
    gross_approval = st.number_input('Amount Desired', min_value=0, max_value=100000)
    sba_guaranteed_approval = st.number_input('SBA Guaranteed Approval if Applicable', min_value=0, max_value=150000)
    initial_interest_rate = st.number_input('Initial Interest Rate Desired', min_value=0.0, max_value=20, value=5.0, format="%.2f")
    term_in_months = st.number_input('Term in Months Desired', min_value=0, max_value=120)
    jobs_supported = st.number_input('Jobs Supported', min_value=0, max_value=1)
    fixed_or_variable_interest = st.selectbox('Interest Type', ['Variable', 'Fixed'])
    business_type_individual = st.radio('Is Individual Business?', ['Yes', 'No'])
    business_type_partnership = st.radio('Is Partnership?', ['Yes', 'No'])

    submitted = st.form_submit_button("Submit")
    if submitted:
        try:
            input_data = np.array([[gross_approval, sba_guaranteed_approval, initial_interest_rate,
                                    term_in_months, jobs_supported,
                                    1 if fixed_or_variable_interest == 'Fixed' else 0,
                                    1 if business_type_individual == 'Yes' else 0,
                                    1 if business_type_partnership == 'Yes' else 0]])

            input_df = pd.DataFrame(input_data, columns=model_columns)
            input_scaled = scaler.transform(input_df)
            prediction = model.predict(input_scaled)
            result = prediction[0][0]

            if result > 0.99:
                st.success('Congratulations, you are approved! A representative will contact you shortly to assist you with your loan request.')
            else:
                st.error('We cannot approve your request at the moment. But we will reach out to help you navigate other options.')
        except Exception as e:
            st.error("An error occurred during the prediction process. Please try again.")
            st.error("Error details: " + str(e))

