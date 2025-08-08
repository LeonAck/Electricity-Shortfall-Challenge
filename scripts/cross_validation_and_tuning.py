from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, RandomizedSearchCV
import omegaconf


def get_search_type(config):
    search_type = config['model_selection']['search_type']
    search_params = config['model_selection']['search_type_params'][search_type]

    if search_type== 'grid':
        RandomizedSearchCV()
        return GridSearchCV(**search_params)
    
    elif search_type== 'random':
        return RandomizedSearchCV(**search_params)
    
    else:
        raise ValueError(f"Unknown search type: {search_type}")
    

def get_search_class_and_params(config):
    """
    Returns the search class and its specific parameters (like n_iter).
    """
    search_type = config['model_selection']['search_type']
    search_params = config['model_selection']['search_type_params'].get(search_type, {})

    if search_type == 'grid':
        return GridSearchCV, {'param_grid': None}  
    elif search_type == 'random':
        return RandomizedSearchCV, {'param_distributions': None, **search_params}
    else:
        raise ValueError(f"Unknown search type: {search_type}")

def get_split_type(config):

    if config['model_selection']['split_type'] == 'time_series':
        return TimeSeriesSplit
    
    else:
        raise ValueError(f"Unknown split type: {config['model_selection']['split_type']}")
    

def get_param_grid(model_config):
    """
    Get the parameter grid for hyperparameter tuning based on the model type.
    """
    
    try:
        return model_config['tuning_params']
       
    except omegaconf.errors.KeyValidationError as e:
        print(model_config)
        raise ValueError(f"Model type {model_config['type']} has no param grid { str(e) }") 
        
