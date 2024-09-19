import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

def show_dataset_info():
    data = {
        'Feature': [
            'CustomerID', 'ProdTaken', 'Age', 'TypeofContact', 'CityTier',
            'DurationOfPitch', 'Occupation', 'Gender', 'NumberOfPersonVisiting',
            'NumberOfFollowups', 'PreferredPropertyStar', 'MaritalStatus',
            'NumberOfTrips', 'Passport', 'PitchSatisfactionScore', 'OwnCar',
            'NumberOfChildrenVisiting', 'Designation', 'MonthlyIncome'
        ],
        'Description': [
            'Unique identifier for each customer.',
            'Indicates if the customer purchased a holiday package (1 = Yes, 0 = No).',
            'Age of the customer.',
            'How the customer was contacted (Company Invited or Self Enquiry).',
            'City tier classification (1, 2, 3), with 1 being the highest tier.',
            'Duration of the sales pitch made to the customer (in minutes).',
            'Customer\'s occupation (e.g., Executive, Manager, VP).',
            'Gender of the customer (Male or Female).',
            'Number of people accompanying the customer on the trip.',
            'Number of follow-up interactions with the customer.',
            'Preferred star rating of the property (1 to 5).',
            'Customer\'s marital status (Married, Single, Divorced).',
            'Number of trips the customer has taken.',
            'Indicates if the customer has a passport (1 = Yes, 0 = No).',
            'Satisfaction score of the sales pitch (1 to 5).',
            'Indicates if the customer owns a car (1 = Yes, 0 = No).',
            'Number of children accompanying the customer on the trip.',
            'Customer\'s job designation (e.g., Executive, Manager, AVP).',
            'Monthly income of the customer.'
        ]
    }
    # Create a DataFrame
    df = pd.DataFrame(data)
    # Display the table in Streamlit
    st.table(df)

def show_dataset_management():
    """
    Creates a donut chart to describe the dataset split composition (calibration, testing, training).
    """
    # Dataset split composition
    total_rows = 4888
    calibration_rows = 489
    testing_rows = 880
    training_rows = total_rows - (calibration_rows + testing_rows)

    labels = ['Calibration', 'Testing', 'Training']
    values = [calibration_rows, testing_rows, training_rows]

    # Create the donut chart
    fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.5)])

    # Customize layout
    fig.update_layout(
        title="Dataset Split Composition",
        annotations=[dict(text='4888 Rows', x=0.5, y=0.5, font_size=20, showarrow=False)],
        showlegend=True
    )

    return st.plotly_chart(fig)

def show_model_performances():
    table = """
    | **Score**      | **CatBoost**                | **RandomForest**             | **XGBoost**                 |
    |----------------|-----------------------------|------------------------------|-----------------------------|
    | **ROC-AUC**    | 88.05 (std 4.77)             | 84.22 (std 1.55)              | 87.13 (std 3.96)             |
    | **PR-AUC**     | 75.60 (std 8.33)             | 69.29 (std 4.28)              | 74.10 (std 10.69)            |
    | **Precision**  | 91.94 (std 5.84)             | 90.86 (std 6.57)              | 90.04 (std 8.21)             |
    | **Recall**     | 77.76 (std 9.66)             | 70.17 (std 3.36)              | 76.02 (std 6.51)             |
    | **F1-Score**   | 83.91 (std 5.96)             | 79.03 (std 2.64)              | 82.82 (std 6.51)             |
    | **Accuracy**   | 94.51 (std 1.89)             | 93.00 (std 1.00)              | 94.07 (std 2.45)             |
    | **Fitting Time**| 20.12 (std 15.16)           | 0.93 (std 0.34)               | 0.39 (std 0.13)              |
    """
    st.write(table)

def show_xgb_model_info():
    st.write("""
    **XGBoost** is a highly efficient and powerful gradient boosting algorithm designed for supervised learning tasks.
    XGBoost consist of *Gradient Boosting* where enables it to Build models sequentially, each correcting the errors of its predecessor that will efficiently works on large datasets with parallel and distributed computing.
    It also support **Regularization** (Incorporates L1 (Lasso) and L2 (Ridge) regularization, and Post-Pruning) to reduce overfitting.
""")

def show_model_architecture():
    content = """
    The model comprises two integrated components: a Pipeline Model featuring an `XGBoostClassifier` as the **Main Model**, and an `Isotonic Regression` model serving as the **Calibration Model**.
    When an input dataset is provided, it first undergoes preprocessing according to the steps defined in the accompanying diagrams. During training, the Feature Balancing step is activated to ensure the model learns from a balanced dataset. However, this step is not applied during prediction.
    Following preprocessing, the dataset is processed by the Core Model, which produces an output referred to as the Internal Output. This Internal Output is then fed into the Calibration Model, which adjusts it to generate the final Calibrated Output."""
    st.write(content)

def show_main_model_performances():
    # Data from the XGBoost column in the image
    categories = ['ROC-AUC', 'PR-AUC', 'Precision', 'Recall', 'f1-Score', 'Accuracy']
    xgboost_scores = [87.13, 74.10, 90.04, 76.02, 82.82, 94.07]

    # Define the spider plot using Plotly
    fig = go.Figure()

    # Add the XGBoost data to the radar chart
    fig.add_trace(go.Scatterpolar(
        r=xgboost_scores,
        theta=categories,
        fill='toself',
        name='XGBoost'
    ))

    # Set plot layout and title
    fig.update_layout(
    polar=dict(
        radialaxis=dict(
        visible=True,
        range=[0, 100]
        ))
    )
    st.plotly_chart(fig)

def member_card(image_path, member_data):
    with st.container(border=True):
        photo_holder, desc_holder = st.columns([1,2])
        photo_holder.image(image_path)
        desc_holder.write(f"#### {member_data['Name']}")
        desc_holder.write(f"{member_data['Job']}")
        desc_holder.caption(f"{member_data['Email']}")

dedi = {
    'Name' : "Dedi Samosir | Dedi",
    'Job' : 'Data Science Tutor',
    'Email' : '-'
}
sulthon = {
    'Name' : "Sulthon Amar | Sulthon",
    'Job' : 'Data Science Lead',
    'Email' : '-'
}
bagas = {
    'Name' : 'Bagas Nur Prayoga | Bagas',
    'Job' : 'Data Science Member',
    'Email' : 'bagasnurprayoga@gmail.com'
}
mia = {
    'Name' : "Mia Listiana | Mia",
    'Job' : 'Data Science Member',
    'Email' : 'mialistiana29@gmail.com'
}
baiq = {
    'Name' : "Zuyyina Hilya | Baiq",
    'Job' : 'Data Science Member',
    'Email' : 'zuyyinahilya56@gmail.com'
}
ayu = {
    'Name' : "Ayu Dahraeni| Ayu",
    'Job' : 'Data Science Member',
    'Email' : '⁠dhrnayu@gmail.com'
}
latief = {
    'Name' : "Latief Al Amien | Latief",
    'Job' : 'Data Science Member',
    'Email' : 'latiefalamien24@gmail.com'
}
nico = {
    'Name' : "Nico Simarmata | Nico",
    'Job' : 'Data Science Member',
    'Email' : 'nicosimarmata88@gmail.com'
}

about_app, about_dataset, about_model, about_team = st.tabs(["About App", "Dataset Info", "Model Info", "Meet Our Team"])

with about_app:
    st.header("Rakamin Data Science Bootcamp Batch 45 Final Project Apps Demo")
    st.image("Pics/cover.jpg")

    st.subheader("1. Project Overview")
    st.image("Pics/1.png")
    st.write("""
    This project aims to predict the likelihood of a customer purchasing a holiday package using machine learning models. By analyzing various customer data points, including demographics and travel history, the app assists "Trips & Travel.Com" in identifying potential buyers for its holiday packages, which include Basic, Standard, Deluxe, Super Deluxe, and King. The focus is on promoting a new **Wellness Tourism Package** while improving the efficiency of marketing strategies.
    """)
    
    st.subheader("2. Problem Statement")
    st.image("Pics/2.png")
    st.write("""
    The company's marketing costs have been high, with only 22% of customers purchasing holiday packages in the last year. One major issue is that marketing campaigns were done randomly, without considering customer profiles. The company now seeks to use data-driven approaches to make their marketing more efficient by targeting the right customers.""")

    st.subheader("3. Goals and Objectives")
    st.image("Pics/3.png")
    st.write("""
    - **Prediction:** Use customer data (such as income, age, occupation, etc.) to predict if they will purchase a holiday package.
    - **Efficiency:** Reduce marketing costs by focusing on high-potential customers.
    - **Insights:** Provide insights on customer segments that are more likely to convert, enabling the marketing team to better target these customers.""")

    ratio = [1,2]
    st.subheader("4. Business Metrics")
    col1, col2 = st.columns(ratio)
    col1.image("Pics/4.png")
    col2.write("""
    **Conversion Rate:** The percentage of customers who purchase a package out of the total number of customers contacted. This metric helps to evaluate the effectiveness of sales efforts. A higher conversion rate indicates that a larger proportion of contacted customers are making purchases.
    **After Using Model**: The Conversion Rate of 1000 samples of dummy production dataset shows that the **increase of Conversion Rate from 21.72 Percent up to 94.62 Percent (3,4X Increase in Conversion Rate).**
    """)
    col3, col4 = st.columns(ratio)
    col3.image("Pics/5.png")
    col4.write("""
    **Customer Acquisition Cost (CAC):** The average cost spent to acquire a new customer who purchases a package. CAC measures the efficiency of the marketing and sales processes. Lowering the CAC while maintaining or increasing customer acquisition is a key goal for businesses.
    **After Using Model**: The CAC of 1000 samples of dummy production dataset shows that the **a Reduction of CPC Cost of 38 Percent.**
    """)
    col5, col6 = st.columns(ratio)
    col5.image("Pics/6.png")
    col6.write("""
    **Contact Rate:** The percentage of customers successfully contacted out of the total potential customers. Contact rate is a measure of outreach effectiveness on how well a business is able to engage with its customer base, which impact sales and conversion rates.
    Calibrated model probabilities provide reliable predictions of customer willingness to buy. This helps Marketing and Sales to optimize strategies by segmenting customers and tailoring engagement.
    
    """)

with about_dataset:
    st.header("Dataset Informations")
    show_dataset_info()

    st.header("Dataset Management")
    show_dataset_management()

with about_model:
    st.header("Core Model Used")
    st.subheader("XGBoost (Extreme Gradient Boosting")
    st.image("Pics/xgb_logo.png", caption="XGBoost Logo")
    show_xgb_model_info()

    st.header("Overall Model Architecture")
    st.image("Pics/model_arch.png", caption="Overall Model Architecture")
    show_model_architecture()

    st.header("Model Performances")
    st.subheader("Main Model Performances")
    show_main_model_performances()
    st.subheader("Main Model Performances Comparisons")
    show_model_performances()
    st.subheader("Model Calibration Results")
    st.image("Pics/calib_results.png", caption="Calibration Results Plot")
    
    st.header("Model Learning Results")
    st.subheader("Feature Importances and SHAP Values")
    st.image("pics/fi_shp (1).png", caption="Feature Importances and SHAP Values")
    st.subheader("Partial Dependences Plot for 5 Best Features")
    st.image("pics/fi_shp (2).png", caption="Partial Dependences Plot of 5 Best Features")
    st.subheader("Partial Dependences Plot for Other Features")
    st.image("pics/fi_shp (3).png", caption="Partial Dependences Plot of Other Features")

with about_team:
    with st.container(border=True):
        st.header("Our Beloved Tutor")
        member_card("Pics/pp_round.png", dedi)
    with st.container(border=True):
        st.header("Our Amazing Teams")
        member_card("Pics/pp_round.png", sulthon)
        member_card("Pics/pp_round.png", bagas)
        member_card("Pics/pp_round.png", mia)
        member_card("Pics/pp_round.png", baiq)
        member_card("Pics/pp_round.png", ayu)
        member_card("Pics/pp_round.png", latief)
        member_card("Pics/pp_round.png", nico)
