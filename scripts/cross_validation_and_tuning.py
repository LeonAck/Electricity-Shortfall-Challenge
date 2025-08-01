from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, RandomizedSearchCV
import omegaconf

from sklearn.metrics import make_scorer, mean_squared_error
import numpy as np

def rmse_scorer(y_true, y_pred):
    """Custom RMSE scorer that handles the squared parameter"""
    rmse_score = np.sqrt(mean_squared_error(y_true, y_pred))
    return make_scorer(
    rmse_score,
    greater_is_better=False  # Lower RMSE is better
    )

def create_rmse_scorer():
    """
    Create a proper RMSE scorer using make_scorer
    """
    def rmse_calculation(y_true, y_pred):
        """Inner function that calculates RMSE"""
        return np.sqrt(mean_squared_error(y_true, y_pred))
    
    # Pass the FUNCTION to make_scorer, not the result
    return make_scorer(
        rmse_calculation,  # ← This is a function, not a value
        greater_is_better=False,  # Lower RMSE is better
        response_method='predict'
    )

def get_search_type(config):

    if config['model_selection']['search_type'] == 'grid':
        return GridSearchCV
    
    elif config['model_selection']['search_type'] == 'random':
        return RandomizedSearchCV
    
    else:
        raise ValueError(f"Unknown search type: {config['model_selection']['search_type']}")
    

def get_split_type(config):

    if config['model_selection']['split_type'] == 'time_series':
        return TimeSeriesSplit
    
    else:
        raise ValueError(f"Unknown split type: {config['model_selection']['split_type']}")
    

def get_param_grid(model, config):
    """
    Get the parameter grid for hyperparameter tuning based on the model type.
    """
    
    try:
        return model['tuning_params']
       
    except omegaconf.errors.KeyValidationError as e:
        raise ValueError(f"Model type {model['type']} has no param grid { str(e) }") 
        print(config['models'])
