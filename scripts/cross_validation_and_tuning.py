from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, RandomizedSearchCV


def get_search_type(config):

    if config['cv_settings']['search_type'] == 'grid':
        return GridSearchCV()
    
    elif config['cv_settings']['search_type'] == 'random':
        return RandomizedSearchCV()
    
    else:
        raise ValueError(f"Unknown search type: {config['cv_settings']['search_type']}")
    

def get_split_type(config):

    if config['cv_settings']['split_type'] == TimeSeriesSplit:
        return TimeSeriesSplit()
    
    else:
        raise ValueError(f"Unknown split type: {config['cv_settings']['split_type']}")
    

def get_param_grid(model, config):
    """
    Get the parameter grid for hyperparameter tuning based on the model type.
    """
    if model['type'] in config['models']:
        return config['models'][model['type']]['param_grid']
    else:
        raise ValueError(f"Model type {model['type']} not found in config.")
    
