import numpy as np
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
from sklearn.base import BaseEstimator, RegressorMixin
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge, SGDRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor 


class DifferencedARIMAWrappedModel:
    def __init__(self, AR_order, difference_order, MA_order):
        self.order = [AR_order, difference_order, MA_order]  # e.g., (1, 1, 0) for first differencing
        self.model = None
        self.last_observed_value = None

    def fit(self, X, y):
        # Store last value to reverse differencing later
        self.last_observed_value = y.iloc[0] if hasattr(y, "iloc") else y[0]

        # Apply first differencing
        y_diff = np.diff(y)

        # Fit ARIMA on differenced series (X is ignored)
        self.model = ARIMA(y_diff, order=(self.order[0], 0, self.order[2])).fit()
        return self

    def predict(self, X):
        # Predict differenced values
        y_pred_diff = self.model.predict(start=1, end=len(X))

        # Reverse differencing (i.e., cumulative sum)
        y_pred = np.r_[self.last_observed_value, y_pred_diff].cumsum()
        return y_pred[1:]  # exclude initial value used for cumsum
    
    def forecast(self, steps):
        y_diff_forecast = self.model.forecast(steps=steps)
        forecast = self.last_observed_value + np.cumsum(y_diff_forecast)
        return forecast

class MA_ModelWrapper:
    def __init__(self, q=2):
        self.q = q
        self.model = None
        self.fitted = None

    def fit(self, y):
        self.model = ARIMA(y, order=(0, self.q, 0))
        self.fitted = self.model.fit()

    def predict(self, start, end):
        return self.fitted.predict(start=start, end=end)
    
    def forecast(self, steps):
        return self.fitted.forecast(steps=steps)
    

class ARModelWrapper:
    def __init__(self, lags=1):
        """
        AR model using ARIMA(order=(p, 0, 0)), where p = number of lags.
        """
        self.lags = lags
        self.model = None
        self.fitted_model = None

    def fit(self, y):
        """
        Fits ARIMA model with order=(lags, 0, 0), i.e., pure AR model.
        """
        self.model = ARIMA(y, order=(self.lags, 0, 0))
        self.fitted_model = self.model.fit()

    def predict(self, start, end):
        """
        Predict from index `start` to `end`.
        """
        return self.fitted_model.predict(start=start, end=end)
    
    def forecast(self, steps):
        return self.fitted.forecast(steps=steps)
    
    
class SklearnModelWrapper:
    def __init__(self, model_cls):
        self.model = model_cls()

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)


def get_model(model_type: str, params: dict):
    if model_type == 'LinearRegression':
        return LinearRegression(**params)

    elif model_type == 'RandomForest':
        return RandomForestRegressor(**params)

    elif model_type == 'Ridge':
        return Ridge(**params)

    elif model_type == 'Lasso':
        return Lasso(**params)

    elif model_type == 'ElasticNet':
        return ElasticNet(**params)

    elif model_type == 'BayesianRidge':
        return BayesianRidge(**params)

    elif model_type == 'SGDRegressor':
        return SGDRegressor(**params)

    elif model_type == 'ExtraTreesRegressor':
        return ExtraTreesRegressor(**params)

    elif model_type == 'GradientBoostingRegressor':
        return GradientBoostingRegressor(**params)

    elif model_type == 'XGBRegressor':  # Add this
        return XGBRegressor(**params)

    elif model_type == 'DecisionTreeRegressor':
        return DecisionTreeRegressor(**params)

    elif model_type == 'AdaBoostRegressor':
        return AdaBoostRegressor(**params)

    elif model_type == 'KNeighborsRegressor':
        return KNeighborsRegressor(**params)

    elif model_type == 'SVR':
        return SVR(**params)

    elif model_type in ['AR', 'MA', 'MA1']:
        return DifferencedARIMAWrappedModel(**params)  # Return AR/MA model wrapper class
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
