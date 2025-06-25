import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
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
        predictions = train_and_predict_ma(model_name, y_train, len(y_val), y_val.index)
        
    else:
        model.fit(X_train, y_train)
        predictions = model.predict(X_val)

    # Bereken RMSE
    rmse = evaluate_model(y_val, predictions)

    plot_predictions(y_val, predictions, model_name, output_dir, dataset_name="validation")
    
    return model, model_name, rmse, predictions

def choose_best_model(output_dir, df, models_to_try, target_column ='load_shortfall_3h'):
    """
    Choose model that scores best on RMSE
    """
    
    # Train en evalueer alle modellen
    print("\nModellen trainen en evalueren op validatieset...")
    results = {}
    best_model = None
    best_rmse = float('inf')

    # split data
    X_train, X_val, y_train, y_val = split_data(df, target_column=target_column, test_size=0.2, random_state=42, time_series=True)
    
    for model_name, model in models_to_try.items():
        print(f"\nEvalueren van {model_name}...")
        trained_model, model_name, rmse, _ = train_and_evaluate_model(output_dir,
            model, model_name, X_train, y_train, X_val, y_val
        )
        results[model_name] = {'model': trained_model, 'rmse': rmse}
        print(f"RMSE: {rmse:.4f}")
        
        if rmse < best_rmse:
            best_rmse = rmse
            best_model = trained_model
            best_model_name = model_name
    
    # Gebruik het beste model voor voorspellingen op de testset
    print(f"\nBeste model: {min(results.items(), key=lambda x: x[1]['rmse'])[0]}")
    print(f"RMSE op validatieset: {best_rmse:.4f}")

    return best_rmse, best_model, best_model_name


def train_full_model_predict_test_set(best_model, train_df, test_df, target_column):

    if best_model == "AutoReg":  # Voor AR1 modellen hebben we alleen y_train nodig
        model_fit, last_value, lags = train_ar_diff_model(train_df[target_column])
        test_predictions = predict_ar_diff(model_fit, last_value, lags, steps=len(test_df), index=test_df.index)

    else:  # Voor andere modellen gebruiken we zowel X_train als y_train
        best_model.fit(train_df.drop(columns=[target_column]), train_df[target_column])
        test_predictions = best_model.predict(test_df)

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


