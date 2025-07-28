import numpy as np
from statsmodels.tsa.arima.model import ARIMA

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
    
