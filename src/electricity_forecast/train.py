from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import make_scorer, root_mean_squared_error
from sklearn.pipeline import Pipeline
from electricity_forecast.models import get_model
from electricity_forecast.preprocessing import get_pipeline_for_model
from electricity_forecast.cross_validation_and_tuning import get_split_type, get_search_class_and_params

import logging
from omegaconf import OmegaConf
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass

import mlflow
from mlflow.tracking import MlflowClient


@dataclass
class ModelResult:
    """Data class to hold model evaluation results"""
    model_type: str
    cv_rmse: float
    best_params: Optional[Dict[str, Any]] = None
    model_config: Optional[Dict[str, Any]] = None


@dataclass
class BestModelResult:
    """Data class to hold final best model selection result"""
    full_pipeline: Any
    model_name: str
    cv_rmse: float
    is_from_mlflow: bool
    production_version: int


# ============================================================================
# CORE TRAINING FUNCTIONS
# ============================================================================


def train_with_cross_validation(model_config: Dict[str, Any], config: Dict[str, Any], 
                               X_train, y_train) -> float:
    """Train model using cross-validation only (no hyperparameter tuning)"""
    model_type = model_config['type']
    model = get_model(model_type, model_config['params'])
    pipeline = get_pipeline_for_model(model_config, config)
    X_train_processed = pipeline.fit_transform(X_train, y_train)

    cv = get_split_type(config)(n_splits=config['model_selection']['n_splits'])
    rmse_scorer = make_scorer(root_mean_squared_error, response_method='predict', greater_is_better=False)

    scores = cross_val_score(model, X_train_processed, y_train, cv=cv, scoring=rmse_scorer)
    logging.debug(f"CV scores for {model_type}: {scores}")
    
    return -np.mean(scores)


def train_with_hyperparameter_tuning(model_config: Dict[str, Any], config: Dict[str, Any], 
                                    X_train, y_train) -> Tuple[float, Dict[str, Any]]:
    """Train model with hyperparameter tuning and cross-validation"""
    model_type = model_config['type']
    pipeline = get_pipeline_for_model(model_config, config)
    model = get_model(model_type, model_config['params'])
    X_train_processed = pipeline.fit_transform(X_train, y_train)
    
    # Get parameter grid
    param_grid = OmegaConf.to_container(model_config['tuning_params'], resolve=True)
    
    if not param_grid or len(param_grid) == 0:
        logging.info(f"No parameters to tune for {model_type}, using cross-validation only")
        cv_rmse = train_with_cross_validation(model_config, config, X_train, y_train)
        return cv_rmse, None
    
    # Setup hyperparameter search
    cv = get_split_type(config)(n_splits=config['model_selection']['n_splits'])
    rmse_scorer = make_scorer(root_mean_squared_error, response_method='predict', greater_is_better=False)
    search_class, search_kwargs = get_search_class_and_params(config)
    
    # Configure search parameters
    param_key = 'param_grid' if search_class == GridSearchCV else 'param_distributions'
    search_kwargs[param_key] = dict(param_grid)
    
    common_params = {
        'estimator': model,
        'cv': cv,
        'scoring': rmse_scorer,
        'n_jobs': config['model_selection'].get('n_jobs', -1),
        'verbose': config['model_selection'].get('verbose', 1)
    }
    
    # Execute search
    search = search_class(**{**search_kwargs, **common_params})
    search.fit(X_train_processed, y_train)
    
    logging.debug(f"Best score for {model_type}: {search.best_score_}")
    return -search.best_score_, search.best_params_


def retrain_full_model(X_train, y_train, config: Dict[str, Any], 
                      model_config: Dict[str, Any], best_params: Optional[Dict[str, Any]] = None) -> Pipeline:
    """Retrain the best model on the full training dataset"""
    # Use best parameters if available, otherwise use default parameters
    params_to_use = best_params if best_params else model_config['params']
    
    model = get_model(model_config['type'], params_to_use)
    pipeline = get_pipeline_for_model(model_config, config)
    X_train_processed = pipeline.fit_transform(X_train, y_train)
    
    model.fit(X_train_processed, y_train)
    
    return Pipeline([
        ('preprocessing', pipeline),
        ('model', model)
    ])


# ============================================================================
# MODEL COMPARISON FUNCTIONS
# ============================================================================

def evaluate_candidate_models(config: Dict[str, Any], X_train, y_train) -> List[ModelResult]:
    """Evaluate all candidate models and return results"""
    candidate_models = []
    
    for model_config in config['models']:
        model_type = model_config['type']
        logging.info(f"Evaluating {model_type}...")
        
        # Train with appropriate method
        if config['model_selection']['tuning']:
            cv_rmse, best_params = train_with_hyperparameter_tuning(model_config, config, X_train, y_train)
        else:
            cv_rmse = train_with_cross_validation(model_config, config, X_train, y_train)
            best_params = None
        
        result = ModelResult(
            model_type=model_type,
            cv_rmse=cv_rmse,
            best_params=best_params,
            model_config=model_config
        )
        
        candidate_models.append(result)
        logging.info(f"{model_type} CV RMSE: {cv_rmse:.4f} with params: {best_params}")
    
    return candidate_models


def get_best_candidate_model(candidate_models: List[ModelResult]) -> ModelResult:
    """Select the best candidate model based on CV RMSE"""
    return min(candidate_models, key=lambda x: x.cv_rmse)


def compare_with_production_model(best_candidate: ModelResult, config: Dict[str, Any]) -> Tuple[bool, Optional[Dict[str, Any]]]:
    """Compare best candidate with existing production model"""
    existing_best = get_best_existing_model(config)
    
    if existing_best and existing_best['cv_rmse'] <= best_candidate.cv_rmse:
        logging.info(f"Existing production model is better (RMSE: {existing_best['cv_rmse']:.4f} vs {best_candidate.cv_rmse:.4f})")
        return True, existing_best
    else:
        logging.info(f"New candidate model is better (RMSE: {best_candidate.cv_rmse:.4f})")
        return False, None


# ============================================================================
# MLFLOW INTEGRATION FUNCTIONS
# ============================================================================

def setup_mlflow_experiment(config: Dict[str, Any]) -> None:
    """Initialize MLflow tracking"""
    mlflow.set_tracking_uri(config['logging']['tracking_uri'])
    mlflow.set_experiment(config['logging']['experiment_name'])


def log_run_metadata(config: Dict[str, Any]) -> None:
    """Log run-level metadata to MLflow"""
    mlflow.set_tag("trigger", config["run"]["trigger"])
    mlflow.set_tag("stage", config["run"]["stage"])
    mlflow.set_tag("data_version", config["data"]["version"])
    mlflow.set_tag("tuning", config['model_selection']['tuning'])


def log_model_evaluation(model_result: ModelResult) -> None:
    """Log individual model evaluation to MLflow"""
    with mlflow.start_run(run_name=model_result.model_type, nested=True):
        mlflow.set_tag("model_type", model_result.model_type)
        
        if model_result.best_params:
            mlflow.log_params(model_result.best_params)
        
        mlflow.log_metric("cv_rmse", model_result.cv_rmse)


def log_candidate_model_evaluations(candidate_models: List[ModelResult]) -> None:
    """Log all candidate model evaluations to MLflow"""
    for model_result in candidate_models:
        log_model_evaluation(model_result)


def log_and_register_final_model(final_model: Pipeline, model_name: str, cv_rmse: float, 
                                config: Dict[str, Any]) -> str:
    """Log and register the final model to MLflow Registry"""
    with mlflow.start_run(run_name="final_best_model", nested=True):
        mlflow.set_tag("final_model_type", model_name)
        mlflow.log_metric("final_cv_rmse", cv_rmse)

        result = mlflow.sklearn.log_model(
            sk_model=final_model,
            name=config["logging"]['object_name'],
            registered_model_name=config['logging']['registered_model_name'], 
            tags={"model_type": model_name} 
        )

        model_version = result.registered_model_version
        promote_model_to_production(config, model_version, model_name)
        
        return model_version


def promote_model_to_production(config: Dict[str, Any], model_version: str, model_name: str) -> None:
    """Promote model version to production alias"""
    client = MlflowClient()
    
    client.set_registered_model_alias(
        name=config['logging']['registered_model_name'],
        version=model_version,
        alias="production"
    )

    client.set_model_version_tag(
        name=config['logging']['registered_model_name'],
        version=model_version,
        key="model_type",
        value=model_name
    )

    logging.info(f"Promoted model version {model_version} to PRODUCTION.")


def get_best_existing_model(config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Retrieve the best existing model from MLflow Registry"""
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

        # Fetch metrics from run
        run_id = version.run_id
        run = client.get_run(run_id)
        cv_rmse = run.data.metrics.get("final_cv_rmse", None)
        model_type = version.tags.get("model_type")
        
        logging.info("Production model loaded successfully")
        return {
            'model': model,
            'model_type': model_type,
            'cv_rmse': float(cv_rmse),
            'run_id': version.run_id,
            'version': version.version
        }

    except Exception as e:
        logging.warning(f"MLflow production model retrieval failed: {str(e)}")
        return None


# ============================================================================
# MAIN ORCHESTRATION FUNCTION
# ============================================================================

def choose_best_model(train_df, config: Dict[str, Any]) -> BestModelResult:
    """
    Main function to choose the best model using cross-validation with MLflow integration
    
    Args:
        output_dir: Directory for output artifacts
        train_df: Training dataframe
        config: Configuration dictionary
    
    Returns:
        BestModelResult with best model and metadata
    """
    # Prepare training data
    y_train = train_df[config['data']['target_column']]
    X_train = train_df.drop(columns=[config['data']['target_column']])
    
    # Setup MLflow
    setup_mlflow_experiment(config)
    
    # Start parent MLflow run
    with mlflow.start_run(run_name="model_selection") as parent_run:
        log_run_metadata(config)
        
        # Evaluate all candidate models
        candidate_models = evaluate_candidate_models(config, X_train, y_train)
        
        # Log all evaluations to MLflow
        log_candidate_model_evaluations(candidate_models)
        
        # Find best candidate from current training run
        best_candidate = get_best_candidate_model(candidate_models)
        
        # Compare with existing production model
        use_existing, existing_best = compare_with_production_model(best_candidate, config)
        
        # Determine final model
        if use_existing:
            final_model = existing_best['model']
            final_rmse = existing_best['cv_rmse']
            final_model_name = existing_best['model_type']
            model_version = existing_best['version']
            is_from_mlflow = True

        else:
            # Train new model on full dataset
            final_model = retrain_full_model(
                X_train, y_train, config, 
                best_candidate.model_config, best_candidate.best_params
            )
            final_rmse = best_candidate.cv_rmse
            final_model_name = best_candidate.model_type
            is_from_mlflow = False
            
            # Log and register new model
            model_version = log_and_register_final_model(
                final_model, final_model_name, final_rmse, config
            )
    
    return BestModelResult(
        full_pipeline=final_model,
        model_name=final_model_name,
        cv_rmse=final_rmse,
        is_from_mlflow=is_from_mlflow,
        production_version=model_version
    )