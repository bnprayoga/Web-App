import streamlit as st
import numpy as np
import pandas as pd
import pickle
import joblib

def compile_manual_input():
    with st.container(border=True):
        st.write("### Basic Informations")

        Age = st.slider("Age", min_value=18, max_value=100, value=(25))        

        colGender, colMaritalStatus  = st.columns([2,4])
        GenderMale = colGender.radio("Gender", ["Male", "Female"], index=None, horizontal=True)
        if GenderMale == "Male": GenderMale = 1
        else: GenderMale = 0
        MaritalStatus = colMaritalStatus.radio("Marital Status", ["Unmarried", "Married", "Divorced"], index=None, horizontal=True)
        
        CityTier, colPassport = st.columns([5,1])
        hasPassport = colPassport.checkbox("Has Passport")
        CityTier = CityTier.selectbox("City Tier", [1, 2, 3], index=None)

    with st.container(border=True):
        st.write("### Job Related Informations")
        
        colDesignation, colIncome = st.columns([1,1])
        monthlyIncome = colIncome.number_input("Monthly Income (US$)", value=20000)
        Designation = colDesignation.selectbox("Designation", 
                            ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
        Occupation = st.radio("Occupations", ["Large Business", "Small Business", "Others"], index=None, horizontal=True)

    with st.container(border=True):
        st.write("### Engagment")
        DurationOfPitch = st.slider("Picth Duration (minutes)", min_value=0, max_value=50, value=(25))
        NumberOfFollowups = st.number_input("Number of Followups", min_value=1, max_value=6, value=3)
        PreferredPropertyStar = st.radio("Preferred Property Star", [3, 4, 5], index=None, horizontal=True)
        st.write("Ratings on Pitch")
        PitchSatisfactionScore = st.feedback("stars")

    row = {
        'Age': [Age],
        'DurationOfPitch' : [DurationOfPitch],
        'NumberOfFollowups': [NumberOfFollowups],
        'PreferredPropertyStar' : [PreferredPropertyStar],
        'MonthlyIncome' : [monthlyIncome],
        'CityTier' : [CityTier],
        'Occupation' : [Occupation],
        'GenderMale' : [GenderMale],
        'MaritalStatus' : [MaritalStatus],
        'HasPassport' : [hasPassport],
        'PitchSatisfactionScore' : [PitchSatisfactionScore],
        'Designation' : [Designation]
    }
    return pd.DataFrame(row)

def check_null(row):
    df_null = row.isna().sum()
    null_col = list(df_null[df_null>0].index)
    return null_col

with open('holiday-package-predictions/Models/final_model_calibrated.pkl', 'rb') as f:
    final_model_calibrated = pickle.load(f)

prediction = None

def infer(prediction):
    if prediction == 1:
        return "Buy"
    else: return "Not Buy"

st.header("Prediction Using Manual Input")
inputs = compile_manual_input()
col1, col2 = st.columns([1,6])
calculate = col1.button("calculate")

if calculate:
    null_cols = check_null(inputs)
    if len(null_cols) > 0:
        st.error(f"There is Empty Columns Left. Total Empty Columns `{len(null_cols)}`")
        st.write("Please fill all of this columns:")
        for col in null_cols:
            st.write(f"- {col}")
    else:
        st.write("Your Input Row")
        st.write(inputs)
        prediction = final_model_calibrated.predict(inputs)
        prediction_prob = final_model_calibrated.predict_proba(inputs)
        col2.success(f"Prediction has been calculated Succesfully")

    if prediction == None:
        st.write("Your Prediction Has Not Been Calculated, Please Submit The Input on Input Tab")
    else:
        st.write(f"### The Probability of Buy: {100*prediction_prob[0][-1] :.2f}%")
        st.write(f"Its likely that The Customer will {infer(prediction)}")
        reset = st.button("Reset Calculation")
        if reset:
            prediction = None
