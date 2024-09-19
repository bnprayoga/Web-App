import streamlit as st
import imblearn
import pickle
import numpy as np
import pandas as pd
pd.set_option("display.max_columns", None)
import matplotlib as mpl
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from imblearn.pipeline import Pipeline, make_pipeline
from imblearn.over_sampling import RandomOverSampler, SMOTENC
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.preprocessing import PowerTransformer, MinMaxScaler, StandardScaler
from sklearn.pipeline import Pipeline as OuterPipeline
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from streamlit_option_menu import option_menu
import about
import plotly.graph_objects as go
import plotly.express as px

class ColumnImputer(TransformerMixin, BaseEstimator):
    """
    Encode Columns
    """
    #initialize function
    def __init__(self,
                 steps=[],
                 remainder='passthrough'):
        
        super().__init__()
        self.steps = steps
        self.remainder = remainder

    #fit somethings
    def fit(self, X, y=None):
        X_train = X.copy()
        self.cl = ColumnTransformer(self.steps,
                                    remainder=self.remainder)
        self.cl.fit(X_train)
        return self

    #transform column
    def transform(self, X, y=None):
        X_test = X.copy()
        
        columns = [col.split('__')[-1] for col in self.cl.get_feature_names_out()]
        
        return pd.DataFrame(self.cl.transform(X_test), columns = columns)

class ICode(TransformerMixin):
    """
    Adding New Features, X must be in Pandas DataFrame format
    """
    #initialize function
    def __init__(self, 
                 ordinal_encoder=None, 
                 nominal_encoder=None,
                 ordinal_column=None,
                 ordinal_column_categories=None,
                 nominal_column=None,
                 nominal_column_categories=None,
                 remainder='passthrough'):
        
        super().__init__()
        self.ordinal_encoder = ordinal_encoder
        self.nominal_encoder = nominal_encoder
        self.ordinal_column = ordinal_column
        self.ordinal_column_categories = ordinal_column_categories
        self.nominal_column = nominal_column
        self.nominal_column_categories  = nominal_column_categories
        self.remainder = remainder

        from sklearn.compose import ColumnTransformer

    #fit somethings
    def fit(self, X, y=None):   
        X_train = X.copy()

        #if only nominal encoder
        if self.ordinal_encoder == None:
            self.ohe = self.nominal_encoder(categories = self.nominal_column_categories,
                                            sparse_output = False)
            self.cl = ColumnTransformer([('ohe', self.ohe, self.nominal_column)], 
                                        remainder=self.remainder)

        #if only ordinal encoder
        elif self.nominal_encoder == None:
            self.oe = self.ordinal_encoder(categories = self.ordinal_column_categories)
            self.cl = ColumnTransformer([('oe', self.oe, self.ordinal_column)], 
                                        remainder=self.remainder)
        
        elif (self.nominal_encoder == None) & (self.ordinal_encoder == None):
            pass

        else:
            self.oe = self.ordinal_encoder(categories = self.ordinal_column_categories)
            self.ohe = self.nominal_encoder(categories = self.nominal_column_categories,
                                            sparse_output = False)
    
            self.cl = ColumnTransformer([('oe', self.oe, self.ordinal_column),
                                         ('ohe', self.ohe, self.nominal_column)], 
                                        remainder=self.remainder)
        self.cl.fit(X_train)
        return self

    #transform column
    def transform(self, X, y=None):
        X_test = X.copy()
        
        columns = [col.split('__')[-1] for col in self.cl.get_feature_names_out()]
        
        return pd.DataFrame(self.cl.transform(X_test), columns = columns).astype('float')


    def inverse_transform(self, X, y=None):
        X = X.copy()
        X = self.cl.inverse_transform(X)
        return X

class TranScaler(TransformerMixin):
    """
    Adding New Features, X must be in Pandas DataFrame format
    """
    #initialize function
    def __init__(self,
                 steps = [],
                 remainder = 'passthrough'):
        
        super().__init__()
        self.steps = steps
        self.remainder = remainder

    #fit somethings
    def fit(self, X, y=None):   
        X_train = X.copy()

        self.cl = ColumnTransformer(self.steps, 
                                    remainder=self.remainder)
        self.cl.fit(X_train)
        
        return self

    #transform column
    def transform(self, X, y=None):
        X_test = X.copy()
        
        columns = [col.split('__')[-1] for col in self.cl.get_feature_names_out()]
        
        return pd.DataFrame(self.cl.transform(X_test), columns = columns)

developer_page = st.Page("about.py", title="Informations")

manual_prediction_page = st.Page("Tools/predict_manual.py", title="Prediction Using Manual Input")

dataset_prediction_page = st.Page("Tools/predict_dataset.py", title="Prediction Using File Input")

#upcoming features
#dataset_page = st.Page("Dataset/dataset_manager.py", title="Get Dataset")
models_page = st.Page("Models/models_manager.py", title="Get Trained Models")

st.set_page_config(
        page_title="Demo App",
        page_icon="holiday-package-predictions/Pics/favicon.png"
)
pg = st.navigation({
    "Home" : [developer_page],
    "Prediction" : [manual_prediction_page, dataset_prediction_page]
})
st.write(numpy.__version__)
st.write(joblib.__version__)
st.write(sklearn.__version__)
st.title("HOLIDAY PACKAGE PREDICTION DEMO APP")
pg.run()
