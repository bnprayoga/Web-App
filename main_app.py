import streamlit as st
from streamlit_option_menu import option_menu
import engines, about
import numpy as np
import matplotlib as mp
import matplotlib.pyplot as plt
from labellines import labelLines

#setting parameter
HEIGHT_START = 120
HEIGHT_END = 250
WEIGHT_START = 15
WEIGHT_END = 300

#create color map for viz styling
my_color_map = {
    "obeseIII" : "#A30000",
    "obeseII" : "#FF3333",
    "obeseI" : "#FF8000",
    "overweight" : "#FFFF00",
    "normal" : "#33FF33",
    "underweight" : "#66FFFF",
    "ex_underweight" : "#BBFAFA"}

tab1, tab2 = st.tabs(["About", "Calculator"])

with tab1:
    #Main BMI Page
    about.Content().intro()
    about.Content().about_BMI()
    about.Content().about_BFP()

with tab2:
    st.header("BMI and BFP Estimator")
    sex = st.radio("Are you Male or Female?", ["Male", "Female"])
    age = st.slider("Age", 19, 70, 25)
    col1, col2 = st.columns(2)
    weight = col1.number_input("Input your Body Weight (kg):", 0)
    height = col2.number_input("Input your Height (cm):", 0)
    calc = st.button("Calculate")
    calc_status = False
    if calc:
        if (weight != 0) and (height != 0):
            BMI = weight/(height/100)**2
            calc_status = True
        else:
            tab2.markdown("Please enter your Weight and Height")

    if calc_status:
        engines.plot_speed_meter_BMI(height, weight, BMI, my_color_map)
        engines.return_user_BMI_class(BMI, calc_status)
        engines.plot_BMI_bar(HEIGHT_START, HEIGHT_END, WEIGHT_START, WEIGHT_END, height, weight, my_color_map)
        engines.show_weight_from_ideal(BMI, weight, height)
        engines.plot_BFP(BMI, age, sex, weight)

    else:
        st.markdown("Please Enter your Weight and Height in Calculation Tab")
