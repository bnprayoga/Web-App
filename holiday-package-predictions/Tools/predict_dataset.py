import streamlit as st
import numpy as np
import pandas as pd
import pickle
import plotly.graph_objects as go
import plotly.express as px

def file_reader(dataset):
    X = pd.read_csv(dataset)
    X.drop(columns='ProdTaken', inplace=True)
    return X

def plot_histogram(y_pred_prob):
    """
    Creates a histogram plot using plotly based on the y_pred_prob values.
    
    Args:
    y_pred_prob (list or array-like): Predicted probabilities.

    Returns:
    A plotly histogram figure.
    """
    # Create a histogram
    fig = go.Figure(data=[go.Histogram(x=y_pred_prob, nbinsx=20)])

    # Set title and labels
    fig.update_layout(
        title="Histogram of Predicted Probabilities",
        xaxis_title="Predicted Probabilities",
        yaxis_title="Count",
        bargap=0.1, # Space between bars
    )
    # Show the plot
    st.plotly_chart(fig)

def plot_donut_chart(y_pred):
    """
    Creates a donut chart using plotly based on the binary y_pred values 
    and displays it in Streamlit. 
    
    Args:
    y_pred (list or array-like): Binary outputs (1 for 'Buy', 0 for 'Not Buy').
    """
    # Count occurrences of 'Buy' (1) and 'Not Buy' (0)
    buy_count = y_pred.sum()
    not_buy_count = len(y_pred) - buy_count

    labels = ['Buy', 'Not Buy']
    values = [buy_count, not_buy_count]

    # Create donut chart
    fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.5)])

    # Customize layout
    fig.update_layout(
        title="Buy vs Not Buy",
        annotations=[dict(text='Buy/Not Buy', x=0.5, y=0.5, font_size=20, showarrow=False)]
    )

    # Show the plot
    st.plotly_chart(fig)

with open('holiday-package-predictions/Models/final_model_calibrated.pkl', 'rb') as f:
    final_model_calibrated = pickle.load(f)

input_tab, report_tab = st.tabs(["Input", "Report"])
prediction = None

with input_tab:
    st.header("Prediction Using Input File")
    with st.expander("Click To Get Templates and Demo Dataset"):
    
        @st.cache_data
        def convert_df(path):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
            df = pd.read_csv(path)
            return df.to_csv(index=False).encode("utf-8")

        template = convert_df("holiday-package-predictions/Dataset/templates.csv")
        dummy = convert_df("holiday-package-predictions/Dataset/dummpy_production_df.csv")
        tempCol, dummyCol = st.columns(2)
        tempCol.download_button(label = "Click To Download Input Template", 
                                data = template,
                                file_name="Template.csv")
        dummyCol.download_button(label = "Click To Download Dummy Dataset", 
                                data = dummy,
                                file_name="DummyDataset.csv")

    inputs = st.file_uploader("Choose Your Input File", type = 'csv')
    output = st.selectbox("Output", ["Probabilites", "Prediction", "Both"], index = 2)
    col1, col2 = st.columns([1,5])
    calculate = col1.button("calculate")
    if inputs:
        X = file_reader(inputs)
        with st.container(border=True):
            st.subheader("Dataset Preview")
            st.dataframe(X)
    if calculate:
        if inputs:
            y_pred_prob = final_model_calibrated.predict_proba(X)[:, -1]
            y_pred = final_model_calibrated.predict(X)
            col2.success("The Calculation Has Completed, Check The Answers in The Reports Tab")
        else: st.error("Please Upload Your Dataset File")

with report_tab:
    if calculate:
        if output == "Probabilites":
            pred_result = pd.concat([X, pd.Series(y_pred_prob)], axis=1)
            pred_result.rename(columns={0: 'Pred_Proba'}, inplace=True)
        elif output == "Prediction":
            pred_result = pd.concat([X, pd.Series(y_pred)], axis=1)
            pred_result.rename(columns={0: 'Pred'}, inplace=True)
        elif output == "Both":
            pred_result = pd.concat([X, pd.Series(y_pred_prob), pd.Series(y_pred)], axis=1)
            pred_result.rename(columns={0: 'Pred', 1 : 'Pred_Proba'}, inplace=True)
        
        with st.expander("CLICK to Show Output DataFrame"):
            st.dataframe(pred_result)

        with st.container(border=True):
            col1, col2 = st.columns(2)
            with col1:
                plot_donut_chart(y_pred)
            with col2:
                plot_histogram(y_pred_prob)

    else: st.write("Your Prediction Has Not Been Calculated, Please Submit The Input on Input Tab")

