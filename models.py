import numpy as np
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
from sklearn.base import BaseEstimator, RegressorMixin
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, LSTM, GRU, Bidirectional, Dense
from sklearn.preprocessing import MinMaxScaler

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge, SGDRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor 

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
    
    # RNN Models
    elif model_type == 'SimpleRNN':
        return RNNRegressor(model_type='SimpleRNN', **params)

    elif model_type == 'LSTM':
        return RNNRegressor(model_type='LSTM', **params)

    elif model_type == 'GRU':
        return RNNRegressor(model_type='GRU', **params)

    elif model_type == 'BiLSTM':
        return RNNRegressor(model_type='BiLSTM', **params)

    elif model_type == 'AR1':
        return "AutoReg"  # Placeholder

    elif model_type in ['MA1', 'MA2', 'SMA']:
        return params  # Return params dict for MA models

    else:
        raise ValueError(f"Unknown model type: {model_type}")
    

class RNNRegressor:
    def __init__(self, model_type='LSTM', units=50, epochs=50, batch_size=32, 
                 sequence_length=24, random_state=42):
        self.model_type = model_type
        self.units = units
        self.epochs = epochs
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.random_state = random_state
        self.model = None
        self.scaler = MinMaxScaler()
        
        # Set random seeds
        tf.random.set_seed(random_state)
        np.random.seed(random_state)
    
    def _create_sequences(self, data, target=None):
        """Create sequences for RNN input"""
        X, y = [], []
        for i in range(self.sequence_length, len(data)):
            X.append(data[i-self.sequence_length:i])
            if target is not None:
                y.append(target[i])
        return np.array(X), np.array(y) if target is not None else None
    
    def _build_model(self, input_shape):
        """Build the RNN model"""
        model = Sequential()
        
        if self.model_type == 'SimpleRNN':
            model.add(SimpleRNN(self.units, input_shape=input_shape))
        elif self.model_type == 'LSTM':
            model.add(LSTM(self.units, input_shape=input_shape))
        elif self.model_type == 'GRU':
            model.add(GRU(self.units, input_shape=input_shape))
        elif self.model_type == 'BiLSTM':
            model.add(Bidirectional(LSTM(self.units), input_shape=input_shape))
        
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model
    
    def fit(self, X, y):
        """Fit the RNN model"""
        # Scale the features
        X_scaled = self.scaler.fit_transform(X)
        
        # Create sequences
        X_seq, y_seq = self._create_sequences(X_scaled, y.values)
        
        if len(X_seq) == 0:
            raise ValueError(f"Not enough data for sequence length {self.sequence_length}")
        
        # Build model
        input_shape = (X_seq.shape[1], X_seq.shape[2])
        self.model = self._build_model(input_shape)
        
        # Train model
        self.model.fit(X_seq, y_seq, 
                      epochs=self.epochs, 
                      batch_size=self.batch_size, 
                      verbose=0,
                      validation_split=0.1)
        
        return self
    
    def predict(self, X):
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model not fitted yet")
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Create sequences
        X_seq, _ = self._create_sequences(X_scaled)
        
        if len(X_seq) == 0:
            # If not enough data for sequences, return mean of last values
            return np.full(len(X), X_scaled[-self.sequence_length:].mean())
        
        # Predict
        predictions = self.model.predict(X_seq, verbose=0).flatten()
        
        # Handle case where we have fewer predictions than requested
        if len(predictions) < len(X):
            # Pad with last prediction
            last_pred = predictions[-1] if len(predictions) > 0 else 0
            padding = np.full(len(X) - len(predictions), last_pred)
            predictions = np.concatenate([padding, predictions])
        
        return predictions
    

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