
import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="PremiumLoan Amount",layout="wide")

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR/"SmartpremiumLoan.pkl"

@st.cache_resource
def load_pipeline():
    return joblib.load(MODEL_PATH)

final_pipeline = load_pipeline()

st.title("Premium Amount Prediction")
st.write("Insurance Premium Prediction Sysytem")
st.write("Enter the details for the prediction of the loan amount")
st.divider()

st.header("Customer Details")
col1,col2 = st.columns(2)
with col1:
    Age = st.number_input("Age",min_value=18,max_value=100,value=32,step=1)
    Gender = st.selectbox("Gender",["Male","Female"])
    Annual_Income = st.number_input("Annual Income",min_value=0.0,max_value=150000.0,value=42000.2,step=500.0)
    Marital_Status = st.selectbox("Marital Status",["Married","Divorced","Single"])
with col2:
    Number_of_dependents = st.number_input("Number of Dependents",min_value=0.0,max_value=4.0,value=1.5,step=0.1)
    Education_level = st.selectbox("Education Level",["Bachelor's","Master's","High School","PhD"])
    Occupation = st.selectbox("Occupation",["Self-Employed","Employed","Unemployed"])
    Location = st.selectbox("Location",["Urban","Rural","Suburban"])

st.header("Health&Lifestyle")
col1,col2=st.columns(2)
with col1:
    Health_score = st.number_input("Health Score",min_value=2.0,max_value=60.0,value=25.2,step=1.0)
    Policy_type = st.selectbox("Policy type",["Premium","Comprehensive","Basic"])
    Vehicle_Age = st.number_input("Vehicle Age",min_value=0.0,max_value=20.0,value=10.2,step=1.0)
with col2:
    Smoking_Status = st.selectbox("Smoking Status",["No","Yes"])
    Exercise_Frequency = st.selectbox("Exercise Frequency",["Weekly","Monthly","Daily","Rarely"])

st.header("Insurance Information")
col1,col2=st.columns(2)
with col1:
    Previous_Claims = st.number_input("Previous Claims",min_value=0.0,max_value=9.0,value=4.5,step=1.0)
    Credit_Score = st.number_input("Credit Score",min_value=300.0,max_value=850.0,value=500.0,step=10.0)
    Insurance_Duration = st.number_input("Insurance Duration",min_value=0.0,max_value=9.0,value=5.6,step=0.5)
with col2:
    Customer_Feedback = st.selectbox("Customer Feedback",["Poor","Average","Good"])
    Property_Type = st.selectbox("Property Type",["House","Apartment","Condo"])
st.divider()

if st.button("Predict Premium",type="primary",use_container_width=True):
    try:
        customer_data = pd.DataFrame({
            "Age": [float(Age)],
            "Gender": [Gender],
            "Annual Income": [float(Annual_Income)],
            "Marital Status": [Marital_Status],
            "Number of Dependents": [float(Number_of_dependents)],
            "Education Level": [Education_level],
            "Occupation": [Occupation],
            "Location": [Location],
            "Health Score": [float(Health_score)],
            "Policy Type": [Policy_type],
            "Vehicle Age": [float(Vehicle_Age)],
            "Smoking Status": [Smoking_Status],
            "Exercise Frequency": [Exercise_Frequency],
            "Previous Claims": [float(Previous_Claims)],
            "Credit Score": [float(Credit_Score)],
            "Insurance Duration": [float(Insurance_Duration)],
            "Customer Feedback": [Customer_Feedback],
            "Property Type": [Property_Type]
        })

        prediction = final_pipeline.predict(customer_data)
        premium = max(0,float(prediction[0]))

        st.success("Premium Amount predicted successfully")
        st.subheader(f"Estimated Insurance Premium: rs{premium:,.2f}")
    
    except Exception as e:
        st.error("An error occured while Generating")
        st.exception(e)
        
