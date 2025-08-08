from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import make_scorer, root_mean_squared_error
from sklearn.pipeline import Pipeline
from scripts.models import get_model
from scripts.preprocessing import get_pipeline_for_model
from scripts.cross_validation_and_tuning import get_split_type, get_param_grid, get_search_class_and_params

import logging
from omegaconf import OmegaConf
import numpy as np

import mlflow
from mlflow.tracking import MlflowClient


def evaluate_model(y_true, y_pred):
    """Calculate RMSE between true and predicted values"""
    return root_mean_squared_error(y_true, y_pred)

def train_model_with_cv(model_config, config, X_train, y_train):
    model_type = model_config['type']
    model = get_model(model_type, model_config['params'])
    pipeline = get_pipeline_for_model(model_config, config)
    X_train_processed = pipeline.fit_transform(X_train, y_train)

    # Get cross-validation type  
    cv = get_split_type(config)(n_splits=config['model_selection']['n_splits'])
    # Create rmse scorer
    rmse_score = make_scorer(root_mean_squared_error, response_method='predict', greater_is_better=False)

    # Perform cross-validation and return average RMSE
    scores = cross_val_score(model, X_train_processed, y_train, cv=cv, scoring=rmse_score)
    print(scores)
    return -np.mean(scores)


def train_model_with_tuning(model_config, config, X_train, y_train):
    """Train a single model with hyperparameter tuning and CV"""
    
    pipeline = get_pipeline_for_model(model_config, config)
    model = get_model(model_config["type"], model_config['params'])
    X_train_processed = pipeline.fit_transform(X_train, y_train)
    param_grid = get_param_grid(model_config)
    param_grid = OmegaConf.to_container(model_config['tuning_params'], resolve=True)

    if not param_grid or len(param_grid) == 0:
        # No parameters to tune - just train with CV
        logging.info(f"No parameters to tune for {model_config['type']}, using cross-validation only")
        empty_params = model_config['params'] 
        return train_model_with_cv(model_config, config, X_train, y_train), empty_params
        
    
    # Get cv type and cv sore
    cv = get_split_type(config)(n_splits=config['model_selection']['n_splits'])
    rmse_score = make_scorer(root_mean_squared_error, response_method='predict', greater_is_better=False)

    # Get search class and its specific params
    search_class, search_kwargs = get_search_class_and_params(config)

    # Set correct param name
    param_key = 'param_grid' if search_class == GridSearchCV else 'param_distributions'
    search_kwargs[param_key] = dict(param_grid)  

    # Shared parameters
    common_params = {
        'estimator': model,
        'cv': cv,
        'scoring': rmse_score,
        'n_jobs': config['model_selection'].get('n_jobs', -1),
        'verbose': config['model_selection'].get('verbose', 1)
    }

    # Combine and instantiate
    search = search_class(**{**search_kwargs, **common_params})

    search.fit(X_train_processed, y_train)
    print(search.best_score_)
    # Return best score and pipeline with best params
    return -search.best_score_, search.best_params_

def get_best_existing_model(config):
    if not config['logging'].get('mlflow_enabled', False):
        return None

    try:
        mlflow.set_tracking_uri(config['logging']['tracking_uri'])
        client = MlflowClient()
        model_name = config['logging']['registered_model_name']

        version = client.get_model_version_by_alias(model_name, "production")
        if not version:
           return None

        model_uri = f"models:/{model_name}/{version.version}"
        model = mlflow.sklearn.load_model(model_uri)

        # ✅ Fetch metrics from run
        run_id = version.run_id
        run = client.get_run(run_id)

        cv_rmse = run.data.metrics.get("final_cv_rmse", None)

        model_type = version.tags.get("model_type")
        print("model loaded successfully")
        return {
            'model': model,
            'model_type': model_type,
            'cv_rmse': float(cv_rmse),
            'run_id': version.run_id,
            'version': version.version
        }

    except Exception as e:
        logging.warning(f"MLflow check failed: {str(e)}")
        return None

def log_model_selection_to_mlflow(model_type, cv_rmse, best_params=None):
    """Log model evaluation to MLflow"""
    
    with mlflow.start_run(run_name=model_type, nested=True):
        mlflow.set_tag("model_type", model_type)

        if best_params:
            mlflow.log_params(best_params)
        
        # Log metrics
        mlflow.log_metric("cv_rmse", cv_rmse)

def retrain_full_model(X_train, y_train, config, model_config, best_params=None):
    """Retrain the best model on full dataset"""
    # For tuned models, we need to extract the base model
    if config['model_selection']['tuning']:
        model = get_model(model_config['type'], best_params if best_params else model_config['params'])
        pipeline = get_pipeline_for_model(model_config, config)
        X_train_processed = pipeline.fit_transform(X_train, y_train)
        model.fit(X_train_processed, y_train)
        return Pipeline([
        ('preprocessing', pipeline),
        ('model', model)
    ])
    else:
        # For non-tuned models
        pipeline = get_pipeline_for_model(model_config, config)
        X_train_processed = pipeline.fit_transform(X_train, y_train)
        model = get_model(model_config['type'], model_config['params'])
        model.fit(X_train_processed, y_train)
        return Pipeline([
        ('preprocessing', pipeline),
        ('model', model)
    ])

def choose_best_model(output_dir, train_df, config):
    """
    Choose best model using cross-validation with MLflow integration
    
    Args:
        output_dir: Directory for output artifacts
        train_df: Training dataframe
        config: Configuration dictionary
    
    Returns:
        Dictionary with best model and metadata
    """
    # Prepare data
    y_train = train_df[config['data']['target_column']]
    X_train = train_df.drop(columns=[config['data']['target_column']])
    best_params = None

    mlflow.set_tracking_uri(config['logging']['tracking_uri'])
    mlflow.set_experiment(config['logging']['experiment_name'])
    
    # Start parent MLflow run
    with mlflow.start_run(run_name="model_selection") as parent_run:
        # Set tags for the run
        mlflow.set_tag("trigger", config["run"]["trigger"])
        mlflow.set_tag("stage", config["run"]["stage"])
        mlflow.set_tag("data_version", config["data"]["version"])
        mlflow.set_tag("tuning", config['model_selection']['tuning'])

        # Evaluate all candidate models
        candidate_models = []
        for model_config in config['models']:
            model_type = model_config['type']
            logging.info(f"Evaluating {model_type}...")
            
            # Train with appropriate method
            if config['model_selection']['tuning']:
                cv_rmse, best_params = train_model_with_tuning(model_config, config, X_train, y_train)
            else:
                cv_rmse = train_model_with_cv(model_config, config, X_train, y_train)

            # Log to MLflow
            log_model_selection_to_mlflow(model_type, cv_rmse, best_params)
            
            candidate_models.append({
                'model_type': model_type,
                'cv_rmse': cv_rmse,
                'best_params': best_params,
                'model_config': model_config
            })
            logging.info(f"  {model_type} CV RMSE: {cv_rmse:.4f} with best estimators { best_params }")
        
        # Find best candidate from current run
        best_candidate = min(candidate_models, key=lambda x: x['cv_rmse'])
        
        # Check MLflow for better existing model AFTER training all candidates
        existing_best = get_best_existing_model(config)
        
        client = MlflowClient()

        # Determine final best model
        if existing_best and existing_best['cv_rmse'] <= best_candidate['cv_rmse']:
            logging.info(f"Using existing MLflow model (RMSE: {existing_best['cv_rmse']:.4f})")
            final_model = existing_best['model']
            final_rmse = existing_best['cv_rmse']
            final_model_name = existing_best['model_type']
            model_version = existing_best['version']
            is_from_mlflow = True

        else:
            logging.info(f"Using newly trained model (RMSE: {best_candidate['cv_rmse']:.4f})")
            # Retrain best candidate on full dataset
            final_model = retrain_full_model(
                X_train, 
                y_train, 
                config, 
                best_candidate['model_config'],
                best_candidate['best_params']
            )
            final_rmse = best_candidate['cv_rmse']
            final_model_name = best_candidate['model_type']
            is_from_mlflow = False

            # Log final model to MLflow Registry
            with mlflow.start_run(run_name="final_best_model", nested=True):
                mlflow.set_tag("final_model_type", final_model_name)
                mlflow.log_metric("final_cv_rmse", final_rmse)

                result = mlflow.sklearn.log_model(
                    sk_model=final_model,
                    name=config["logging"]['object_name'],  # For backward compatibility
                    registered_model_name=config['logging']['registered_model_name'], 
                    tags={"model_type": final_model_name} 
                )

                model_version = result.registered_model_version

                # Promote this new version to production
                client.set_registered_model_alias(
                    name=config['logging']['registered_model_name'],
                    version=model_version,
                    alias="production"
                )

                client.set_model_version_tag(
                    name=config['logging']['registered_model_name'],
                    version=model_version,
                    key="model_type",
                    value=final_model_name
    
                )

                logging.info(f"Promoted model version {model_version} to PRODUCTION.")
    
    # Prepare results
    return {
        "full_pipeline": final_model,
        "model_name": final_model_name,
        "cv_rmse": final_rmse,
        "is_from_mlflow": is_from_mlflow,
        'production_version': model_version,
    }


