import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from plots import plot_predictions
from models import get_model
from preprocessing import get_pipeline_for_model

import os
import joblib


def split_data(X_train: np.array, y_train: np.array, train_val_split=0.2):
    """
    Split the data into training and validation sets.
    
    Args:
        X_train: Features for training
        y_train: Target variable
        train_val_split: Proportion of data to use for validation
    """

    train_val_loc = int(len(X_train) * (1 - train_val_split))

    X_train_new = X_train[:train_val_loc]
    X_val = X_train[train_val_loc:]

    y_train_new = y_train[:train_val_loc]
    y_val = y_train[train_val_loc:]

    return X_train_new, X_val, y_train_new, y_val


def choose_best_model(output_dir, train_df, config, train_val_split=0.2):
    best_rmse = float("inf")
    best_model = None
    best_model_name = ""
    best_X_train = None
    best_pipeline = None

    y_train = train_df[config['data']['target_column']]
    X_train = train_df.drop(columns=[config['data']['target_column']])
    
    for model in config['models']:

        pipeline = get_pipeline_for_model(model, config)
        X_train_processed = pipeline.fit_transform(X_train, y_train)
        
        y_train = np.array(y_train)

        X_train_new, X_val, y_train_new, y_val = split_data(X_train_processed, y_train, train_val_split)

        model_type = get_model(model['type'], model['params'])
        trained_model = model_type.fit(X_train_new, y_train_new)

        # Predict, evaluate and plot
        predictions = trained_model.predict(X_val)
        rmse = evaluate_model(y_val, predictions)
        plot_predictions(y_val, predictions, model['type'], output_dir, dataset_name="validation")
        
        print(f"Model: {model['type']}, RMSE: {rmse:.4f}")

        if rmse < best_rmse:
            best_rmse = rmse
            best_model = model
            best_model_name = model['type']
            best_X_train = X_train
            best_pipeline = pipeline

    print(f"Best Model: {best_model_name}, Best RMSE: {best_rmse:.4f}")

    best_model_results = {
        "model_object": best_model,
        "pipeline": best_pipeline,
        "model_name": best_model_name,
        "X_train": best_X_train,
        "rmse": best_rmse
    }
    
    return best_model_results


def evaluate_model(y_true, y_pred):
    """
    Bereken de Root Mean Squared Error
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))
