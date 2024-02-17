import streamlit as st
import pandas as pd

text_intro = """
        Once considered a high-income country problem, overweight and obesity are now on the rise in low- and middle-income countries, particularly in urban settings. This fenomena might occured due to pace of rapid tech improvement that shift us to the sedentary lifestyle.  
        Obesity is a condition of abnormal or excess fat accumulation in adipose tissue, to the extent that health may be impaired. At present decades, more than 1 billion (1 in 8 people or nearly 13%) people are considered as obese worldwide; among them 650 million are adults, 
        340 million are adolescents, and 39 million are children. It is estimated that by 2030, globally 1 in 5 women and 1 in 7 men will have obesity. Early evaluation of obesity and overweight status is necessary to prevent and control obesity and overweight associated with 
        some diseases, such as: premature death, cardiovascular diseases, high blood pressure, osteoarthritis, some cancers and also diabetes. Based on aformention facts, it is paramount for individuals to maintain of their healthy body-weight status. The simplest way to measure body-weight
        status is by using BMI and BFP estimation.
"""
text_about_BMI = """
        BMI or Body Mass Index is a measurement of a person's leanness or corpulence that has been discovered by Lambert Adolphe Jacque Quetele in 1874. BMI calculation is based on the measurement of body height and weight that intended to quantify tissue mass. It is widely used as a general indicator of whether a person has a healthy body weight compared their height. 
        Specifically, the value obtained from the BMI calculation yields person categories of either underweight, normal weight, overweight, or obese depending on what range the value falls in between. The BMI itself is defined as the body mass divided by the square of the body height, and is expressed in units of kg.m^-2 as shown on the equation below:
"""

text_about_BFP = """
        BFP or Body Fat Precentage, as shown by its name, is just an indicator that tells us about the magnitude of fraction of fat within our body. Body fat is necessary to store lipids from which the body generates energy. It also secretes a bunch of number of important hormones, and provides the body with some cushioning as well as insulation. 
        Total body fat consists of Essential body fat (EBF) and Storage Body Fat (SBF), both of them are vital to good health. EBF is the minimum amount of fat necessary for normal physiological function that is invinsibly constructed deep inside the body, meanwhile SBF accumulated within adipose tissue, in the form subcutaneous fat that is commonly found under the dermis and wrapped around vital organs, such as around the liver, pancreas, heart, intestines, and kidneys. 
        Too much SBF, especially in waist area, increases risk of various non-communicable diseases. On the other hand, when SBF becomes too little; causes various problems, such as difficulties in temperature regulation, hunger, fatigue, depression, women infertility, etc. Below is the BFP formula by BMI method: 
"""

BMI_dict = {
    "Classification" : ['Extreme Underweight','Underweight', 'Ideal', 'Overweight', 'Obese Type I', 'Obese Type II', 'Obese Type III'],
    "BMI Range" : ['<16.5','16.5-18.5', '18.5-25', '25-30', '30-35', '35-40', '>40'],
    "Risk of Co-morbidities" : ['High (for other desease)','Low (for other desease)', 'Average', 'Increased', 'Moderate', 'Severe', 'Very Severe']
}

BFP_dict = {
    "Description" : ['Essential Fats','Athletes', 'Fitness', 'Acceptable', 'Obese'],
    "BFP of Woman" : ['10-13','14-20', '21-24', '25-31', '>=32'],
    "BFP of Man" : ['2-5','6-13', '14-17', '18-24', '>=25']
}

BMI_table = pd.DataFrame(BMI_dict)
BFP_table = pd.DataFrame(BFP_dict)



class Content():
    def intro(self):
        st.header("Problem with Obesity")
        st.image("body-status/images/obese.jpg", use_column_width = "always")
        st.markdown('<div style="text-align: justify;">{}</div>'.format(text_intro), unsafe_allow_html=True)
        st.markdown('---')

    def about_BMI(self):
        st.header("What is BMI?")
        st.image("body-status/images/BMI.jpg", use_column_width = "always")
        st.markdown('<div style="text-align: justify;">{}</div>'.format(text_about_BMI), unsafe_allow_html=True)
        st.latex(r"""BMI = \frac{weight \left( kg \right)} {height \left( m^2 \right)}""")
        st.markdown('<h5 style="text-align: center;">{}</h5>'.format("BMI Table"), unsafe_allow_html=True)
        st.dataframe(BMI_table, use_container_width=True, hide_index=True)
        st.markdown('---')

    def about_BFP(self):
        st.header("What is BFP?")
        st.image("body-status/images/fat.png", use_column_width = "always")
        st.markdown('<div style="text-align: justify;">{}</div>'.format(text_about_BFP), unsafe_allow_html=True)
        st.latex(r"""BFP = 1.2*BMI + 0.23*age - 10.8*gender - 5.4""")
        st.markdown("""* **Note** that this method is not really accurate to estimate BFP, but for the simplicity, we will utilizeBMI method as a fancy way to get rough estimate of BFP""")
        st.markdown('<h5 style="text-align: center;">{}</h5>'.format("BFP Table"), unsafe_allow_html=True)
        st.dataframe(BFP_table, use_container_width=True, hide_index=True)
        st.markdown('---')
