import numpy as np
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
from sklearn.base import BaseEstimator, RegressorMixin
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

AVAILABLE_MODELS = {
    'AR1': 'Autoregressive Model (AR1)',
    'LinearRegression': 'Linear Regression',
    'RandomForest': 'Random Forest'
}

def get_models():
    return {
        'AR1': "AutoReg",  # AR1 model is handled separately       
        'LinearRegression': LinearRegression(),
        'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42)
    }

def train_ar_diff_model(y_train, lag=2):
    """Train AR model on differenced y_train."""
    diff_train = y_train.diff().dropna()
    model = AutoReg(diff_train, lags=lag, old_names=False)
    model_fit = model.fit()
    
    return model_fit, y_train.iloc[-1], diff_train[-lag:]  # return last value & lags


def predict_ar_diff(model_fit, last_train_value, initial_lags, steps, index):
    """
    Predicts using a fitted AR model on differenced data and returns forecast in original scale.
    
    Parameters:
    - model_fit: fitted AR model (on differenced data)
    - last_train_value: last actual value before prediction period
    - initial_lags: last `lag` differenced values from training data
    - steps: number of time steps to forecast (len(y_val) or len(y_test))
    - index: target index for predictions (index of y_val or y_test)
    """
    pred_diff = []
    history = list(initial_lags)  # last `lag` values used to bootstrap

    coef = model_fit.params.values
    intercept = coef[0]
    lags_coef = coef[1:]
    lag = len(lags_coef)

    for t in range(steps):
        lagged_values = history[-lag:][::-1]
        yhat_diff = intercept + np.dot(lags_coef, lagged_values)
        pred_diff.append(yhat_diff)
        history.append(yhat_diff)

    # Reverse differencing
    pred_original = [last_train_value + pred_diff[0]]
    for i in range(1, steps):
        pred_original.append(pred_original[-1] + pred_diff[i])

    return pd.Series(pred_original, index=index)


#def get_models():
 #   return {
  #      'AR1': AR1Model(),
   #     'LinearRegression': LinearRegression(),
    #    'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42)
    #}


class AR1Model(BaseEstimator):
    def fit(self, y):
        self.model = AutoReg(y, lags=1).fit()
        return self

    def predict(self, X):
        # Forecasting the next len(X) steps based on the trained AR model
        return self.model.predict(start=len(self.model.model.endog), 
                                  end=len(self.model.model.endog) + len(X) - 1)

class MAModel(BaseEstimator, RegressorMixin):
    def __init__(self, order=1):
        self.order = order
        self.model = None
        
    def fit(self, y):
        """
        Train een MA model
        """
        self.model = ARIMA(y, order=(0, 0, self.order))
        self.model = self.model.fit()
        return self
        
    def predict(self, X):
        """
        Maak voorspellingen voor het aantal opgegeven stappen
        """
        if self.model is None:
            raise ValueError("Model moet eerst getraind worden")
        steps = len(X)
        return self.model.forecast(steps=steps) 