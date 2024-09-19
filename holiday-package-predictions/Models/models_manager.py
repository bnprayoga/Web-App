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
import warnings
warnings.filterwarnings('ignore')

seed = 6

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
# Get all obejct
nan_col = ['Age',
           'DurationOfPitch',
           'NumberOfFollowups',
           'PreferredPropertyStar',
           'MonthlyIncome']
imputer = ColumnImputer([('median_imputer',
                          SimpleImputer(strategy='median'),
                          nan_col)])

# Feature Balancing Step
cat_col_smotenc = [
    'PreferredPropertyStar',
    'CityTier',
    'Occupation',
    'MaritalStatus',
    'HasPassport',
    'Designation',
    'GenderMale'
]
cat_col = [
    'PreferredPropertyStar',
    'CityTier',
    'Occupation',
    'MaritalStatus',
    'Designation'
]
cat_categories = [
    [3.0,4.0,5.0],
    [1,2,3],
    ['Others','Small Business','Large Business'],
    ['Unmarried','Married','Divorced'],
    ['Executive','Manager','Senior Manager','AVP','VP']
]
ros = RandomOverSampler(random_state=seed)

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
encoder_lb = ICode(ordinal_encoder = OrdinalEncoder,
                   ordinal_column = cat_col,
                   ordinal_column_categories = cat_categories)

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
num_col = ['Age', 'DurationOfPitch', 'NumberOfFollowups', 'PitchSatisfactionScore', 'MonthlyIncome']
transcaler = TranScaler([('ptf', PowerTransformer(), num_col)])


with open('Models/final_model_calibrated.pkl', 'rb') as f:
    final_model_calibrated = pickle.load(f)

st.success("model loaded successfully")