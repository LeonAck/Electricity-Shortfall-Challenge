import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from plots import plot_predictions
from models import train_ar_diff_model, predict_ar_diff



def split_data(df, target_column, test_size=0.2, random_state=42, time_series=False):
    X = df.drop(columns=[target_column])
    y = df[target_column]

    if time_series:
        split_index = int(len(df) * (1 - test_size))
        return X.iloc[:split_index], X.iloc[split_index:], y.iloc[:split_index], y.iloc[split_index:]
    else:
        return train_test_split(X, y, test_size=test_size, random_state=random_state)

def evaluate_model(y_true, y_pred):
    """
    Bereken de Root Mean Squared Error
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))

def train_and_evaluate_model(output_dir, model, model_name, X_train, y_train, X_val, y_val):
    """ Train en evalueer een model op de validatieset. """

    if model_name == "AR1":
        model_fit, last_value, lags = train_ar_diff_model(y_train)
        predictions = predict_ar_diff(model_fit, last_value, lags, steps=len(y_val), index=y_val.index)
        
    elif model_name in ["MA1", "MA2", "SMA"]:
        predictions = train_and_predict_ma(model_name, y_train, len(y_val), y_val.index if isinstance(y_val, pd.Series) else None)
        
    else:
        model.fit(X_train, y_train)
        predictions = model.predict(X_val)

    # Bereken RMSE
    rmse = evaluate_model(y_val, predictions)

    plot_predictions(y_val, predictions, model_name, output_dir, dataset_name="validation")

    return model, model_name, rmse, predictions

def choose_best_model(output_dir, y_train, models_to_try, train_val_split=0.2):
    best_rmse = float("inf")
    best_model = None
    best_model_name = ""
    best_X_train = None
    best_X_test = None
    
    for model_name, entry in models_to_try.items():
        model = entry['model']
        X_train = entry['X_train']
        X_test = entry['X_test']
        
        # Define split point
        train_val_loc = int(len(X_train) * (1-train_val_split))

        # Split
        X_train_new = X_train[:train_val_loc]
        X_val = X_train[train_val_loc:]

        y_train_new = y_train.iloc[:train_val_loc]
        y_val = y_train.iloc[train_val_loc:]

        # Here: simple train on X_train, evaluate on same data (adjust to proper CV or split if needed)
        model, model_name, rmse, predictions = train_and_evaluate_model(
            output_dir, model, model_name, X_train_new, y_train_new, X_val, y_val
        )
        
        print(f"Model: {model_name}, RMSE: {rmse:.4f}")

        if rmse < best_rmse:
            best_rmse = rmse
            best_model = model
            best_model_name = model_name
            best_X_train = X_train
            best_X_test = X_test

    print(f"Best Model: {best_model_name}, Best RMSE: {best_rmse:.4f}")
    
    return best_rmse, best_model, best_model_name, best_X_train, best_X_test


def train_full_model_predict_test_set(best_model, X_train, X_test, y_train):

    if best_model == "AutoReg":  # Voor AR1 modellen hebben we alleen y_train nodig
        model_fit, last_value, lags = train_ar_diff_model(y_train)
        test_predictions = predict_ar_diff(model_fit, last_value, lags, steps=len(X_test), index=X_test.index if isinstance(X_test, pd.DataFrame) else None)

    elif best_model in ["MA1", "MA2", "SMA"]:
        test_predictions = train_and_predict_ma(best_model, y_train, len(X_test), X_test.index if isinstance(X_test, pd.DataFrame) else None)

    else:  # Voor andere modellen gebruiken we zowel X_train als y_train
        best_model.fit(X_train, y_train.values)
        test_predictions = best_model.predict(X_test)

    return test_predictions

def train_and_predict_ma(model_name, y_train, prediction_steps, y_val_index=None):
    """
    Train and predict using Moving Average models
    
    Args:
        model_name: 'MA1', 'MA2', or 'SMA'
        y_train: Training time series
        prediction_steps: Number of steps to predict
        y_val_index: Index for predictions
    """
    predictions = []
    
    if model_name == 'MA1':
        # MA(1) - uses last residual
        # For simplicity, using naive implementation
        last_value = y_train.iloc[-1]
        for _ in range(prediction_steps):
            pred = last_value  # Simplified MA(1)
            predictions.append(pred)
            
    elif model_name == 'MA2':
        # MA(2) - uses last 2 residuals
        last_values = y_train.tail(2).mean()
        for _ in range(prediction_steps):
            pred = last_values  # Simplified MA(2)
            predictions.append(pred)
            
    elif model_name == 'SMA':
        # Simple Moving Average
        window = min(8, len(y_train))  # 24-hour window or available data
        sma_value = y_train.tail(window).mean()
        predictions = [sma_value] * prediction_steps
    
    # Convert to pandas Series with proper index
    if y_val_index is not None:
        return pd.Series(predictions, index=y_val_index)
    else:
        return np.array(predictions)


