from typing import List, Dict
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder


class WeekdayImputer(BaseEstimator, TransformerMixin):
    """ Impute missing values in weekday column by extracting dayname from date column """

    def __init__(self, date_variable: str, weekday_variable: str):
        if not isinstance(date_variable, str):
            raise ValueError("date_variable should be a str")
        if not isinstance(weekday_variable, str):
            raise ValueError("weekday_variable should be a str")
        
        self.date_variable = date_variable
        self.weekday_variable = weekday_variable

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        return self

    def transform(self, X: pd.DataFrame):
        X = X.copy()
        wkday_null_idx = X[X[self.weekday_variable].isnull() == True].index
        X.loc[wkday_null_idx, self.weekday_variable] = X.loc[wkday_null_idx, self.date_variable].dt.day_name().apply(lambda x: x[:3])

        return X


class WeatherImputer(BaseEstimator, TransformerMixin):
    """ Impute missing values in weather column by replacing them with the most frequent category value """

    def __init__(self, variable: str):
        if not isinstance(variable, str):
            raise ValueError("variable should be a str")
        
        self.variable = variable

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        self.fill_value = X[self.variable].mode()[0]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X[self.variable] = X[self.variable].fillna(self.fill_value)

        return X


class Mapper(BaseEstimator, TransformerMixin):
    """
    Ordinal categorical variable mapper:
    Treat column as Ordinal categorical variable, and assign values accordingly
    """

    def __init__(self, variable: str, mappings: Dict):
        if not isinstance(variable, str):
            raise ValueError("variable should be a str")

        if not isinstance(mappings, dict):
            raise ValueError("mappings should be a dict")

        self.variable = variable
        self.mappings = mappings

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X[self.variable] = X[self.variable].map(self.mappings).astype(int)

        return X


class OutlierHandler(BaseEstimator, TransformerMixin):
    """
    Change the outlier values:
        - to upper-bound, if the value is higher than upper-bound, or
        - to lower-bound, if the value is lower than lower-bound respectively.
    """

    def __init__(self, variable: str):
        if not isinstance(variable, str):
            raise ValueError("variable should be a str")
        
        self.variable = variable

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        q1 = X.describe()[self.variable].loc['25%']
        q3 = X.describe()[self.variable].loc['75%']
        iqr = q3 - q1
        self.lower_bound = q1 - (1.5 * iqr)
        self.upper_bound = q3 + (1.5 * iqr)

        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for i in X.index:
            if X.loc[i,self.variable] > self.upper_bound:
                X.loc[i,self.variable] = self.upper_bound
            if X.loc[i,self.variable] < self.lower_bound:
                X.loc[i,self.variable] = self.lower_bound

        return X


class WeekdayOneHotEncoder(BaseEstimator, TransformerMixin):
    """ One-hot encode weekday column """

    def __init__(self, variable: str):
        if not isinstance(variable, str):
            raise ValueError("variable should be a str")
        
        self.variable = variable

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        self.encoder = OneHotEncoder(sparse_output=False)
        self.encoder.fit(X[[self.variable]])

        self.enc_wkday_features = self.encoder.get_feature_names_out([self.variable])

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X[self.enc_wkday_features] = self.encoder.transform(X[[self.variable]])
        X = X.drop(labels=[self.variable], axis=1)

        return X


class UnusedColumnsDropper(BaseEstimator, TransformerMixin):
    """ Drop unused columns """

    def __init__(self, variables):
        if not isinstance(variables, List):
            raise ValueError("variables should be a list")
        
        self.variables = variables

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        return self

    def transform(self, X: pd.DataFrame):
        X = X.copy()
        X = X.drop(labels=self.variables, axis=1)

        return X
