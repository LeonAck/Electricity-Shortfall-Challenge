import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from plots import plot_predictions
from statsmodels.tsa.ar_model import AutoReg
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

def train_and_evaluate_model(model, name, X_train, y_train, X_val, y_val): 
    """ Train en evalueer een model op de validatieset. Checkt of het een AR1Model is en past de training daarop aan. """ 
    # Check model type en train het model op de juiste manier 

    if name == "AR1":  # Voor AR1 modellen hebben we alleen y_train nodig
        model_fit, last_value, lags = train_ar_diff_model(y_train)
        predictions = predict_ar_diff(model_fit, last_value, lags, steps=len(y_val), index=y_val.index)

    else:  # Voor andere modellen gebruiken we zowel X_train als y_train
        model.fit(X_train, y_train)
        predictions = model.predict(X_val)

    # Bereken RMSE
    rmse = evaluate_model(y_val, predictions)
    print(y_val)
    #plot_predictions(y_val, predictions, dataset_name="validation")
    
    return model, rmse, predictions

def choose_best_model(df, models_to_try, target_column ='load_shortfall_3h'):
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
    
    for name, model in models_to_try.items():
        print(f"\nEvalueren van {name}...")
        trained_model, rmse, _ = train_and_evaluate_model(
            model, name, X_train, y_train, X_val, y_val
        )
        results[name] = {'model': trained_model, 'rmse': rmse}
        print(f"RMSE: {rmse:.4f}")
        
        if rmse < best_rmse:
            best_rmse = rmse
            best_model = trained_model
    
    # Gebruik het beste model voor voorspellingen op de testset
    print(f"\nBeste model: {min(results.items(), key=lambda x: x[1]['rmse'])[0]}")
    print(f"RMSE op validatieset: {best_rmse:.4f}")

    return best_rmse, best_model


def train_full_model_predict_test_set(best_model, train_df, test_df, target_column):

    if best_model == "AutoReg":  # Voor AR1 modellen hebben we alleen y_train nodig
        model_fit, last_value, lags = train_ar_diff_model(train_df[target_column])
        test_predictions = predict_ar_diff(model_fit, last_value, lags, steps=len(test_df), index=test_df.index)

    else:  # Voor andere modellen gebruiken we zowel X_train als y_train
        best_model.fit(train_df.drop(columns=[target_column]), train_df[target_column])
        test_predictions = best_model.predict(test_df)

    return test_predictions