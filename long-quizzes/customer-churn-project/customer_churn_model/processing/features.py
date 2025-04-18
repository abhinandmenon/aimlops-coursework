from typing import List, Dict
import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder


class NewFeaturesComputer(BaseEstimator, TransformerMixin):
    """ Compute additional numerical features for the customer churn dataset """

    def __init__(self):
        pass

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        return self

    def transform(self, X: pd.DataFrame):
        X = X.copy()
        X['Tenure by Age'] = X['Tenure'] / X['Age']
        X['Support Calls by Tenure'] = X['Support Calls'] / X['Tenure']
        X['Payment Delay by Tenure'] = X['Payment Delay'] / X['Tenure']

        return X


class CustomOneHotEncoder(BaseEstimator, TransformerMixin):
    """ One-hot encode categorical column """

    def __init__(self, variable: str):
        if not isinstance(variable, str):
            raise ValueError("variable should be a str")

        self.variable = variable

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        self.encoder = OneHotEncoder(dtype=np.int64, sparse_output=False)
        self.encoder.fit(X[[self.variable]])

        self.encoded_features = self.encoder.get_feature_names_out([self.variable])

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X[self.encoded_features] = self.encoder.transform(X[[self.variable]])
        X = X.drop(labels=[self.variable], axis=1)

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
