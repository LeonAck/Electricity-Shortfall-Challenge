import numpy as np
from sklearn.metrics import mean_squared_error
from scripts.plots import plot_predictions
from scripts.models import get_model
from scripts.preprocessing import get_pipeline_for_model

# goed kijken hoe dit zit
def split_data(X_train: np.array, y_train: np.array, train_val_split=0.2):
    """
    Split the data into training and validation sets.
    
    Args:
        X_train: Features for training
        y_train: Target variable
        train_val_split: Proportion of data to use for validation
    """

    train_val_loc = int(len(X_train) * (1 - train_val_split))

    X_train_new = X_train.iloc[:train_val_loc]
    X_val = X_train.iloc[train_val_loc:]

    y_train_new = y_train.iloc[:train_val_loc]
    y_val = y_train.iloc[train_val_loc:]

    return X_train_new, X_val, y_train_new, y_val


def choose_best_model(output_dir, train_df, config):
    """
    Choose the best model based on RMSE from the training and validation set.
    Args:
        output_dir: Directory to save results and plots
        train_df: DataFrame containing training data
        config: Configuration dictionary containing model parameters and settings
        train_val_split: Proportion of data to use for validation
    """
    best_rmse = float("inf")
    best_model = None
    best_model_name = ""
    best_model_config = None

    y_train = train_df[config['data']['target_column']]
    X_train = train_df.drop(columns=[config['data']['target_column']])

    X_train_new, X_val, y_train_new, y_val = split_data(X_train, y_train, config['preprocessing']['train_val_split'])
    
    y_train_new = np.array(y_train_new)
    y_val = np.array(y_val)

    for model in config['models']:

        pipeline = get_pipeline_for_model(model, config)
        X_train_processed = pipeline.fit_transform(X_train_new)

        model_type = get_model(model['type'], model['params'])
        trained_model = model_type.fit(X_train_processed, y_train_new)

        X_val_processed = pipeline.transform(X_val)

        # Predict, evaluate and plot
        predictions = trained_model.predict(X_val_processed)
        rmse = evaluate_model(y_val, predictions)

        if config['output']['plots']:
            plot_predictions(y_val, predictions, model['type'], output_dir, dataset_name="validation")
        
        print(f"Model: {model['type']}, RMSE: {rmse:.4f}")

        if rmse < best_rmse:
            best_rmse = rmse
            best_model = trained_model
            best_model_name = model['type']
            best_model_config = model

    print(f"Best Model: {best_model_name}, Best RMSE: {best_rmse:.4f}")

    # Retrain on full training set
    full_pipeline = get_pipeline_for_model(best_model_config, config)
    X_full_processed = full_pipeline.fit_transform(X_train, y_train)
    trained_model_full = best_model.fit(X_full_processed, y_train)

    best_model_results = {
        "model_object": trained_model_full,
        "pipeline": full_pipeline,
        "model_name": best_model_name,
        "rmse": best_rmse
    }
    
    return best_model_results


def evaluate_model(y_true, y_pred):
    """
    Calculate root mean squared error (RMSE) between true and predicted values.
    Args:
        y_true: True target values
        y_pred: Predicted values from the model
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))
