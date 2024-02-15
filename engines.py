import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

def plot_BMI_bar(HEIGHT_START, HEIGHT_END, WEIGHT_START, WEIGHT_END, height, weight, my_color_map):
    user_weight_status = [int(i*(height/100)**2) for i in [16.5, 18.5, 25, 30, 35, 40, 2*weight/(height/100)**2]]
    plot, ax = plt.subplots(1,1, figsize=(10, 1))
    plt.title("Your Current Weight Position")
    ax.barh("b", user_weight_status[0], color=my_color_map[list(my_color_map.keys())[::-1][0]], left=0)
    ax.barh("b", user_weight_status[1], color=my_color_map[list(my_color_map.keys())[::-1][1]], left=user_weight_status[0])
    ax.barh("b", user_weight_status[2], color=my_color_map[list(my_color_map.keys())[::-1][2]], left=user_weight_status[1])
    ax.barh("b", user_weight_status[3], color=my_color_map[list(my_color_map.keys())[::-1][3]], left=user_weight_status[2])
    ax.barh("b", user_weight_status[4], color=my_color_map[list(my_color_map.keys())[::-1][4]], left=user_weight_status[3])
    ax.barh("b", user_weight_status[5], color=my_color_map[list(my_color_map.keys())[::-1][5]], left=user_weight_status[4])
    ax.barh("b", user_weight_status[6], color=my_color_map[list(my_color_map.keys())[::-1][6]], left=user_weight_status[5])
    ax.set_xlim(15, 2*weight)
    ax.set_xticks(user_weight_status)
    ax.get_yaxis().set_visible(False)
    ax.set_xlabel("weight (kg)")
    ax.scatter(weight, "b")
    plot.legend(["your position", 
                "extremely underweight",
                "underweight",
                "ideal",
                "overweight",
                "obese type I",
                "obese type II",
                "obese type III"], loc="lower left", bbox_to_anchor=(0.11, -2, 1, 1))
    return st.write(plot)

def show_weight_from_ideal(BMI, weight, height):
    upb_val = weight-24*(height/100)**2
    lwb_val = weight-19.5*(height/100)**2
    if BMI > 25:
        st.success("##### Lose {:.2f} to {:.2f} kg to get Ideal Weight!!!".format(upb_val,lwb_val))
    elif BMI < 18.5:
        st.success("##### Gain {:.2f} to {:.2f} kg to get Ideal Weight!!!".format(-lwb_val, -upb_val))
    else:
        st.success("##### Congratulation, your weight is Ideal")

def create_line(i, n):
    return n*(i/100)**2

def return_user_BMI_class(BMI, calc_status):
    if calc_status:
        if BMI >= 40:
            st.subheader("You are considered as Obese type III, Extemely Obese!")
        elif BMI >= 35:
            st.subheader("You are considered as Obese type II")
        elif BMI >= 30:
            st.subheader("You are considered as Obese type II")
        elif BMI >= 25:
            st.subheader("You are considered as Overweight")
        elif BMI >= 18.5:
            st.subheader("Congratulation, you are in ideal body weight range")
        elif BMI < 18.5:
            st.subheader("You are considered as Underweight")

def plot_speed_meter_BMI(height, weight, BMI, my_color_map):
    st.markdown("## Your Computed BMI")
    fig = go.Figure(go.Indicator(domain = {'x': [0, 1], 'y': [0, 1]}, value = BMI, mode = "gauge+number+delta", 
                                delta = {'reference': 22.75},
                                gauge = {'axis': {'range': [None, 2*weight/(height/100)**2]},
                                        'bar' : {'color' : 'darkslategrey',
                                                'thickness' : 0.2},
                                        'steps' : [{'range': [0, 16.5], 'color': my_color_map[list(my_color_map.keys())[::-1][0]]},
                                                {'range': [16.5, 18.5], 'color': my_color_map[list(my_color_map.keys())[::-1][1]]},
                                                {'range': [18.5, 25], 'color': my_color_map[list(my_color_map.keys())[::-1][2]]},
                                                {'range': [25, 30], 'color': my_color_map[list(my_color_map.keys())[::-1][3]]},
                                                {'range': [30, 35], 'color': my_color_map[list(my_color_map.keys())[::-1][4]]},
                                                {'range': [35, 40], 'color': my_color_map[list(my_color_map.keys())[::-1][5]]},
                                                {'range': [40, 2*weight/(height/100)**2], 'color': my_color_map[list(my_color_map.keys())[::-1][6]]}],
                                        'axis' : {'tickmode' : 'array',
                                                'tickvals' : [16.5, 18.5, 25, 30, 35, 40],
                                                'ticktext' : ["16.5", "18.5", "25", "30", "35", "40"]}}))
    st.plotly_chart(fig)

def plot_BFP(BMI, age, sex, weight):
    if sex == 'Male':
        BFP = 1.2*BMI + 0.23*age - 10.8 - 5.4
    elif sex == 'Female':
        BFP = 1.2*BMI + 0.23*age - 5.4

    labels = ['Fat Mass (kg)', 'Lean Mass (kg)']
    values = [weight*(BFP/100), (1-(BFP/100))*weight]
    colors = ["gold", "mediumturquoise"]

    fig = go.Figure(data=[go.Pie(labels=labels, values=values, 
                                marker=dict(colors=colors), pull=[0.1, 0], 
                                hole=.25)])
    fig.update_layout(title={'text':'Your Body Fat Rough Estimation', 'x':0.5, 'y':0.9, 'xanchor':'center', 'yanchor':'top'},
                    showlegend = False)
    fig.update_traces(textinfo='label+percent+value', textfont_size=15)

    #st.markdown("####  Your Fat Rough Estimation")
    st.plotly_chart(fig)
