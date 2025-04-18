from typing import List, Dict
import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder


class RatioFeaturesComputer(BaseEstimator, TransformerMixin):
    """ Compute additional numerical features for the customer churn dataset """

    def __init__(self, variables: Dict[str, str]):
        if not isinstance(variables, dict):
            raise ValueError("variables should be a dict")
        if not all(isinstance(k, str) for k in variables.keys()):
            raise ValueError("all keys (numerator variables) should be str")
        if not all(isinstance(v, str) for v in variables.values()):
            raise ValueError("all values (denominator variables) should be str")
        
        self.variables = variables
        
    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        return self

    def transform(self, X: pd.DataFrame):
        X = X.copy()

        for num_var, denom_var in self.variables.items():
            X[num_var + ' by ' + denom_var] = X[num_var] / X[denom_var]

        return X


class CustomOneHotEncoder(BaseEstimator, TransformerMixin):
    """ One-hot encode categorical column """

    def __init__(self, variables: List[str]):
        if not isinstance(variables, list):
            raise ValueError("variables should be a list")
        if not all(isinstance(v, str) for v in variables):
            raise ValueError("all values in list (categorical features) should be str")

        self.variables = variables

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        self.encoders = dict()
        self.encoded_features = dict()

        for var in self.variables:
            encoder = OneHotEncoder(dtype=np.int64, sparse_output=False)
            encoder.fit(X[[var]])

            self.encoders[var] = encoder
            self.encoded_features[var] = encoder.get_feature_names_out([var])

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        for var in self.variables:
            X[self.encoded_features[var]] = self.encoders[var].transform(X[[var]])
            X = X.drop(labels=[var], axis=1)

        return X


class UnusedColumnsDropper(BaseEstimator, TransformerMixin):
    """ Drop unused columns """

    def __init__(self, variables):
        if not isinstance(variables, list):
            raise ValueError("variables should be a list")

        self.variables = variables

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        return self

    def transform(self, X: pd.DataFrame):
        X = X.copy()
        X = X.drop(labels=self.variables, axis=1)

        return X
