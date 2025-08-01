import numpy as np
import mlflow
from mlflow.tracking import MlflowClient
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from scripts.models import get_model
from scripts.preprocessing import get_pipeline_for_model
from scripts.cross_validation_and_tuning import get_search_type, get_split_type, get_param_grid, rmse_scorer, create_rmse_scorer
import logging

def evaluate_model(y_true, y_pred):
    """Calculate RMSE between true and predicted values"""
    return np.sqrt(mean_squared_error(y_true, y_pred))

def train_model_with_cv(model_config, config, X_train, y_train):
    model_type = model_config['type']
    model = get_model(model_type, model_config['params'])
    pipeline = get_pipeline_for_model(model_config, config)
    
    # Create full pipeline: preprocessing + model
    full_pipeline = Pipeline([
        ('preprocessing', pipeline),
        ('model', model)
    ])
    
    cv = get_split_type(config)(n_splits=config['model_selection']['n_splits'])

    # Define RMSE scorer
    rmse_scorer = make_scorer(mean_squared_error, squared=False)

    # Perform cross-validation and return average RMSE
    scores = cross_val_score(full_pipeline, X_train, y_train, cv=cv, scoring=rmse_scorer)
    print(scores)
    return np.mean(scores), pipeline


def train_model_with_tuning(model_config, config, X_train, y_train):
    """Train a single model with hyperparameter tuning and CV"""
    pipeline = get_pipeline_for_model(model_config, config)
    model = get_model(model_config["type"], model_config['params'])
    X_train_processed = pipeline.fit_transform(X_train, y_train)
    param_grid = get_param_grid(model_config, config)

    if not param_grid or len(param_grid) == 0:
        # No parameters to tune - just train with CV
        logging.info(f"No parameters to tune for {model_config['type']}, using cross-validation only")
        return train_model_with_cv(model_config, config, X_train, y_train)
    
    # Get search type (GridSearchCV or RandomizedSearchCV)
    search_type = get_search_type(config)
    cv = get_split_type(config)(n_splits=config['model_selection']['n_splits'])
    rmse = create_rmse_scorer()
    # Perform hyperparameter search
    search = search_type(
        estimator=model,
        param_grid=dict(param_grid),
        cv=cv,
        scoring=rmse_scorer,
        n_jobs=1,
        verbose=config['model_selection'].get('verbose', 1)
    )
    search.fit(X_train_processed, y_train)
    print(search.best_score_)
    # Return best score and pipeline with best params
    return np.sqrt(search.best_score_), search.best_estimator_

def get_best_existing_model(config):
    """Check MLflow for existing better models"""
    if not config['logging'].get('mlflow_enabled', False):
        return None
    
    try:
        mlflow.set_tracking_uri(config['logging']['tracking_uri'])
        experiment = mlflow.get_experiment_by_name(config['logging']['experiment_name'])
        if not experiment:
            return None
            
        client = MlflowClient()
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="metrics.cv_rmse > 0",
            run_view_type="ACTIVE_ONLY",
            max_results=1,
            order_by=["metrics.cv_rmse ASC"]
        )
        
        if runs:
            run_id = runs[0].info.run_id
            model_uri = f"runs:/{run_id}/model"
            model = mlflow.sklearn.load_model(model_uri)
            
            return {
                'model': model,
                'model_type': runs[0].data.params.get('model_type', 'unknown'),
                'cv_rmse': runs[0].data.metrics['cv_rmse'],
                'run_id': run_id
            }
    
    except Exception as e:
        logging.warning(f"MLflow check failed: {str(e)}")
        return None
    
    return None

def log_model_to_mlflow(model_type, cv_rmse, config, model_config=None, pipeline=None):
    """Log model evaluation to MLflow"""
    if not config['logging'].get('mlflow_enabled', False):
        return
    
    mlflow.set_tracking_uri(config['logging']['tracking_uri'])
    mlflow.set_experiment(config['logging']['experiment_name'])
    
    with mlflow.start_run():
        # Log model type and parameters
        mlflow.log_param("model_type", model_type)
        if model_config:
            for param, value in model_config['params'].items():
                mlflow.log_param(param, value)
        
        # Log metrics
        mlflow.log_metric("cv_rmse", cv_rmse)
        
        # Log the pipeline/model
        if pipeline:
            mlflow.sklearn.log_model(pipeline, "model")

def retrain_full_model(model, X_train, y_train, config, model_config):
    """Retrain the best model on full dataset"""
    # For tuned models, we need to extract the base model
    if config['model_selection']['tuning']:
        pipeline = model
        pipeline.fit(X_train, y_train)
        return pipeline
    else:
        # For non-tuned models
        pipeline = get_pipeline_for_model(model_config, config)
        X_train_processed = pipeline.fit_transform(X_train, y_train)
        model = get_model(model_config['type'], model_config['params'])
        model.fit(X_train_processed, y_train)
        return pipeline

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
    
    # Evaluate all candidate models
    candidate_models = []
    for model_config in config['models']:
        model_type = model_config['type']
        logging.info(f"Evaluating {model_type}...")
        
        # Train with appropriate method
        if config['model_selection']['tuning']:
            cv_rmse, pipeline = train_model_with_tuning(model_config, config, X_train, y_train)
        else:
            cv_rmse, pipeline = train_model_with_cv(model_config, config, X_train, y_train)
        
        # Log to MLflow
        log_model_to_mlflow(model_type, cv_rmse, config, model_config, pipeline)
        
        candidate_models.append({
            'model_type': model_type,
            'cv_rmse': cv_rmse,
            'pipeline': pipeline,
            'model_config': model_config
        })
        logging.info(f"  {model_type} CV RMSE: {cv_rmse:.4f}")
    
    # Find best candidate from current run
    best_candidate = min(candidate_models, key=lambda x: x['cv_rmse'])
    
    # Check MLflow for better existing model AFTER training all candidates
    existing_best = get_best_existing_model(config)
    
    # Determine final best model
    if existing_best and existing_best['cv_rmse'] < best_candidate['cv_rmse']:
        logging.info(f"Using existing MLflow model (RMSE: {existing_best['cv_rmse']:.4f})")
        final_model = existing_best['model']
        final_rmse = existing_best['cv_rmse']
        final_model_name = existing_best['model_type']
        is_from_mlflow = True
    else:
        logging.info(f"Using newly trained model (RMSE: {best_candidate['cv_rmse']:.4f})")
        # Retrain best candidate on full dataset
        final_model = retrain_full_model(
            best_candidate['pipeline'], 
            X_train, 
            y_train, 
            config, 
            best_candidate['model_config']
        )
        final_rmse = best_candidate['cv_rmse']
        final_model_name = best_candidate['model_type']
        is_from_mlflow = False
    
    # Prepare results
    return {
        "model_object": final_model,
        "pipeline": final_model,
        "model_name": final_model_name,
        "cv_rmse": final_rmse,
        "is_from_mlflow": is_from_mlflow
    }


