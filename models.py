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


class SMA_ModelWrapper:
    def __init__(self, window=3):
        self.window = window
        self.y = None

    def fit(self, y):
        self.y = y

    def predict(self, start, end):
        preds = []
        for i in range(start, end + 1):
            if i < self.window:
                preds.append(np.mean(self.y[:i]))  # partial window
            else:
                preds.append(np.mean(self.y[i - self.window:i]))
        return np.array(preds)

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

    elif model_type == 'AR':
        return ARModelWrapper(**params)  # Return AR model wrapper class

    elif model_type == 'MA':
        return MA_ModelWrapper(**params)  # Return MA model wrapper class
    
    elif model_type == 'SMA':
        return SMA_ModelWrapper(**params)  # Return SMA model wrapper class
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
