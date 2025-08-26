import os
import sys
import pytest
import joblib
import hydra
import numpy as np
from omegaconf import DictConfig
from pathlib import Path
from unittest.mock import patch, MagicMock
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline

<<<<<<< HEAD
from src.data_loading import load_data
from src.config_and_logging import load_config_hydra
from src.train import (
=======
from electricity_forecast.data_loading import load_data
from electricity_forecast.config_and_logging import load_config_hydra
from electricity_forecast.train import (
>>>>>>> f8f64ecb666768cf0d9ec227bc7ae180a7defcb5
    train_with_cross_validation,
    train_with_hyperparameter_tuning,
    retrain_full_model,
    evaluate_candidate_models,
    get_best_candidate_model,
    compare_with_production_model,
    get_best_existing_model,
    choose_best_model,
    ModelResult,
    BestModelResult
)


# Get the project root directory
project_root = Path(__file__).parent.parent
config_path = project_root / "configs"


@pytest.fixture
def config(config_name="config_cv.yaml"):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config directory not found at {config_path}")
    return load_config_hydra(config_name=config_name, config_path=str(config_path))


@pytest.fixture
def test_config(config_name="config_test.yaml"):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config directory not found at {config_path}")
    return load_config_hydra(config_name=config_name, config_path=str(config_path))


@pytest.fixture
def train_and_test_df(config):
    train_df, test_df, _ = load_data(config)
    return train_df, test_df


# ========================================
# Unit Tests Using Your Fixtures
# ========================================

def test_load_config_and_data(config, train_and_test_df):
    train_df, _ = train_and_test_df
    assert 'run' in config
    assert 'models' in config
    assert 'logging' in config
    assert 'output' in config
    assert 'data' in config
    assert 'preprocessing' in config
    assert 'model_selection' in config
    assert not train_df.empty
    assert config['data']['target_column'] in train_df.columns



<<<<<<< HEAD
@patch("scripts.train.get_model")
=======
@patch("electricity_forecast.train.get_model")
>>>>>>> f8f64ecb666768cf0d9ec227bc7ae180a7defcb5
def test_train_with_cross_validation(
    mock_get_model,          
    config, train_and_test_df):

    train_df, _ = train_and_test_df

    X_train = train_df.drop(columns=[config['data']['target_column']])
    y_train = train_df[config['data']['target_column']]

    model_config = config['models'][0]  # First model in config

    # Mock model
    mock_model = LinearRegression()

    mock_model.fit = MagicMock(return_value=mock_model)
    mock_model.predict = MagicMock(return_value=y_train.values)  # dummy pred
    mock_get_model.return_value = mock_model

    cv_rmse = train_with_cross_validation(model_config, config, X_train, y_train)

    print(f"Mocks set up. get_model: {mock_get_model}")
    print("get_model calls:", mock_get_model.call_count)


    assert isinstance(cv_rmse, float)
    assert cv_rmse > 0
    mock_get_model.assert_called()


<<<<<<< HEAD
@patch("scripts.train.GridSearchCV")
=======
@patch("electricity_forecast.train.GridSearchCV")
>>>>>>> f8f64ecb666768cf0d9ec227bc7ae180a7defcb5
def test_train_with_hyperparameter_tuning_no_params(mock_grid_search,
                                                    config, train_and_test_df):
    train_df, _ = train_and_test_df
    X_train = train_df.drop(columns=[config['data']['target_column']])
    y_train = train_df[config['data']['target_column']]

    # Model without tuning params
    model_config = config["models"][0]
    model_config["tuning_params"] = {}
    
    # Mock GridSearchCV
    mock_search = MagicMock()
    mock_search.best_score_ = -0.45
    mock_search.best_params_ = {"alpha": 1.0}
    mock_grid_search.return_value = mock_search

<<<<<<< HEAD
    with patch("scripts.train.train_with_cross_validation") as mock_cv:
=======
    with patch("electricity_forecast.train.train_with_cross_validation") as mock_cv:
>>>>>>> f8f64ecb666768cf0d9ec227bc7ae180a7defcb5
        mock_cv.return_value = 0.123
        cv_rmse, best_params = train_with_hyperparameter_tuning(model_config, config, X_train, y_train)
        assert cv_rmse == 0.123
        assert best_params is None


<<<<<<< HEAD
@patch("scripts.train.get_search_class_and_params")
@patch("scripts.train.get_split_type")
@patch("scripts.train.make_scorer")
@patch("scripts.train.get_pipeline_for_model")
@patch("scripts.train.get_model")
=======
@patch("electricity_forecast.train.get_search_class_and_params")
@patch("electricity_forecast.train.get_split_type")
@patch("electricity_forecast.train.make_scorer")
@patch("electricity_forecast.train.get_pipeline_for_model")
@patch("electricity_forecast.train.get_model")
>>>>>>> f8f64ecb666768cf0d9ec227bc7ae180a7defcb5
def test_train_with_hyperparameter_tuning_targeted(
    mock_get_model,
    mock_get_pipeline,
    mock_make_scorer,
    mock_get_split_type,
    mock_get_search_class_and_params,
    config,
    train_and_test_df
):
    """Target the specific components used in the function"""
    train_df, _ = train_and_test_df
    X_train = train_df.drop(columns=[config['data']['target_column']])
    y_train = train_df[config['data']['target_column']]
    
    model_config = config['models'][1]
    
    # Mock all the helper functions
    mock_model = Ridge(alpha=1.0, random_state=42)
    mock_get_model.return_value = mock_model
    
    # Mock pipeline but return real numpy arrays
    mock_pipeline = MagicMock()
    n_samples, n_features = X_train.shape
    mock_X_processed = np.random.randn(n_samples, n_features)  # Realistic processed data
    mock_pipeline.fit_transform.return_value = mock_X_processed
    mock_get_pipeline.return_value = mock_pipeline
    
    # Mock the search components
    mock_cv = MagicMock()
    mock_get_split_type.return_value.return_value = mock_cv
    
    mock_scorer = MagicMock()
    mock_make_scorer.return_value = mock_scorer
    
    # Mock the search class - this is the key part
    mock_search_class = MagicMock()
    mock_search_instance = MagicMock()
    
    # Configure the search instance to return our desired values
    mock_search_instance.fit.return_value = mock_search_instance
    mock_search_instance.best_score_ = -0.45  # Negative because function negates it
    mock_search_instance.best_params_ = {"alpha": 0.1}
    
    mock_search_class.return_value = mock_search_instance
    mock_get_search_class_and_params.return_value = (mock_search_class, {"param_grid": {}})
    
    # Run the function
<<<<<<< HEAD
    from src.train import train_with_hyperparameter_tuning
=======
    from electricity_forecast.train import train_with_hyperparameter_tuning
>>>>>>> f8f64ecb666768cf0d9ec227bc7ae180a7defcb5
    cv_rmse, best_params = train_with_hyperparameter_tuning(model_config, config, X_train, y_train)
    
    # Assertions - function returns -search.best_score_
    assert cv_rmse == 0.45, f"Expected 0.45, got {cv_rmse}"
    assert best_params == {"alpha": 0.1}
    
    # Verify the search was called correctly
    mock_search_instance.fit.assert_called_once_with(mock_X_processed, y_train)


<<<<<<< HEAD
@patch("scripts.train.get_pipeline_for_model")
@patch("scripts.train.get_model")
=======
@patch("electricity_forecast.train.get_pipeline_for_model")
@patch("electricity_forecast.train.get_model")
>>>>>>> f8f64ecb666768cf0d9ec227bc7ae180a7defcb5
def test_retrain_full_model_with_best_params(
    mock_get_model,
    mock_get_pipeline,
    config,
    train_and_test_df
):
    """Test retrain_full_model using mocked components and provided best params"""
<<<<<<< HEAD
    from src.train import retrain_full_model
=======
    from electricity_forecast.train import retrain_full_model
>>>>>>> f8f64ecb666768cf0d9ec227bc7ae180a7defcb5

    train_df, _ = train_and_test_df
    X_train = train_df.drop(columns=[config['data']['target_column']])
    y_train = train_df[config['data']['target_column']]

    model_config = config['models'][1]
    best_params = {"alpha": 0.5}

    # Mock the model
    mock_model = MagicMock(spec=Ridge)
    mock_get_model.return_value = mock_model

    # Mock the pipeline
    mock_pipeline = MagicMock()
    n_samples, n_features = X_train.shape
    mock_X_processed = np.random.randn(n_samples, n_features)
    mock_pipeline.fit_transform.return_value = mock_X_processed
    mock_get_pipeline.return_value = mock_pipeline

    # Call the function
    result_pipeline = retrain_full_model(X_train, y_train, config, model_config, best_params)

    # Assertions
    mock_get_model.assert_called_once_with(model_config['type'], best_params)
    mock_get_pipeline.assert_called_once_with(model_config, config)
    mock_pipeline.fit_transform.assert_called_once_with(X_train, y_train)
    mock_model.fit.assert_called_once_with(mock_X_processed, y_train)

    # The returned object should be a Pipeline with preprocessing and model
    assert isinstance(result_pipeline, Pipeline)
    assert result_pipeline.steps[0][0] == "preprocessing"
    assert result_pipeline.steps[0][1] == mock_pipeline
    assert result_pipeline.steps[1][0] == "model"
    assert result_pipeline.steps[1][1] == mock_model


<<<<<<< HEAD
@patch("scripts.train.train_with_hyperparameter_tuning")
@patch("scripts.train.train_with_cross_validation")
=======
@patch("electricity_forecast.train.train_with_hyperparameter_tuning")
@patch("electricity_forecast.train.train_with_cross_validation")
>>>>>>> f8f64ecb666768cf0d9ec227bc7ae180a7defcb5
def test_evaluate_candidate_models_with_tuning(
    mock_train_cv,
    mock_train_tuning,
    config,
    train_and_test_df
):
    """Test evaluate_candidate_models when hyperparameter tuning is enabled"""
<<<<<<< HEAD
    from src.train import evaluate_candidate_models
=======
    from electricity_forecast.train import evaluate_candidate_models
>>>>>>> f8f64ecb666768cf0d9ec227bc7ae180a7defcb5

    # Arrange
    train_df, _ = train_and_test_df
    X_train = train_df.drop(columns=[config['data']['target_column']])
    y_train = train_df[config['data']['target_column']]

    config['model_selection']['tuning'] = True

    # Mock tuning return values
    mock_train_tuning.side_effect = [
        (0.25, {"alpha": 0.1}),
        (0.35, {"max_depth": 5})
    ]

    # Act
    results = evaluate_candidate_models(config, X_train, y_train)

    # Assert
    assert isinstance(results, list)
    assert all(isinstance(r, ModelResult) for r in results)
    assert results[0].model_type == config['models'][0]['type']
    assert results[0].cv_rmse == 0.25
    assert results[0].best_params == {"alpha": 0.1}
    assert results[1].cv_rmse == 0.35
    assert results[1].best_params == {"max_depth": 5}

    # Ensure tuning was called for each model
    assert mock_train_tuning.call_count == len(config['models'])
    mock_train_cv.assert_not_called()


def test_get_best_candidate_model_returns_lowest_cv_rmse():
    """Should return the model with the smallest cv_rmse."""
<<<<<<< HEAD
    from src.train import get_best_candidate_model
=======
    from electricity_forecast.train import get_best_candidate_model
>>>>>>> f8f64ecb666768cf0d9ec227bc7ae180a7defcb5

    # Arrange
    results = [
        ModelResult(model_type="A", cv_rmse=0.6),
        ModelResult(model_type="B", cv_rmse=0.4),
        ModelResult(model_type="C", cv_rmse=0.5),
    ]

    # Act
    best = get_best_candidate_model(results)

    # Assert
    assert isinstance(best, ModelResult)
    assert best.model_type == "B"
    assert best.cv_rmse == 0.4


<<<<<<< HEAD
@patch("scripts.train.get_best_existing_model")
=======
@patch("electricity_forecast.train.get_best_existing_model")
>>>>>>> f8f64ecb666768cf0d9ec227bc7ae180a7defcb5
def test_compare_with_production_model_prefers_existing(
    mock_get_existing,
    config,
    train_and_test_df
):
    """If existing model CV RMSE is better, should prefer it."""
<<<<<<< HEAD
    from src.train import compare_with_production_model
=======
    from electricity_forecast.train import compare_with_production_model
>>>>>>> f8f64ecb666768cf0d9ec227bc7ae180a7defcb5

    # Arrange
    mock_get_existing.return_value = {"cv_rmse": 0.3}

    train_df, _ = train_and_test_df
    y_train = train_df[config['data']['target_column']]

    candidate = ModelResult(
        model_type="new_model",
        cv_rmse=0.35,
        best_params={},
        model_config={}
    )

    # Act
    use_existing, existing = compare_with_production_model(candidate, config)

    # Assert
    mock_get_existing.assert_called_once_with(config)
    assert use_existing is True
    assert existing == {"cv_rmse": 0.3}



<<<<<<< HEAD
@patch("scripts.train.get_best_existing_model")
=======
@patch("electricity_forecast.train.get_best_existing_model")
>>>>>>> f8f64ecb666768cf0d9ec227bc7ae180a7defcb5
def test_compare_with_production_model_prefers_new(
    mock_get_existing,
    config
):
    """If candidate model CV RMSE is better, should use it instead of existing."""
<<<<<<< HEAD
    from src.train import compare_with_production_model
=======
    from electricity_forecast.train import compare_with_production_model
>>>>>>> f8f64ecb666768cf0d9ec227bc7ae180a7defcb5

    # Arrange
    mock_get_existing.return_value = {"cv_rmse": 0.5}

    candidate = ModelResult(
        model_type="new_model",
        cv_rmse=0.4,
        best_params={},
        model_config={}
    )

    # Act
    use_existing, existing = compare_with_production_model(candidate, config)

    # Assert
    mock_get_existing.assert_called_once_with(config)
    assert use_existing is False
    assert existing is None


<<<<<<< HEAD
@patch("scripts.train.mlflow.log_metric")
@patch("scripts.train.mlflow.log_param")
@patch("scripts.train.mlflow.log_params")
@patch("scripts.train.mlflow.set_tag")
@patch("scripts.train.mlflow.start_run")
@patch("scripts.train.setup_mlflow_experiment")
@patch("scripts.train.evaluate_candidate_models")
@patch("scripts.train.get_best_candidate_model")
@patch("scripts.train.compare_with_production_model")
=======
@patch("electricity_forecast.train.mlflow.log_metric")
@patch("electricity_forecast.train.mlflow.log_param")
@patch("electricity_forecast.train.mlflow.log_params")
@patch("electricity_forecast.train.mlflow.set_tag")
@patch("electricity_forecast.train.mlflow.start_run")
@patch("electricity_forecast.train.setup_mlflow_experiment")
@patch("electricity_forecast.train.evaluate_candidate_models")
@patch("electricity_forecast.train.get_best_candidate_model")
@patch("electricity_forecast.train.compare_with_production_model")
>>>>>>> f8f64ecb666768cf0d9ec227bc7ae180a7defcb5
def test_choose_best_model_prefers_existing(
    mock_compare,
    mock_get_best,
    mock_evaluate,
    mock_setup_mlflow,
    mock_start_run,
    mock_set_tag,
    mock_log_params,
    mock_log_param,
    mock_log_metric,
    config,
    train_and_test_df
):
<<<<<<< HEAD
    from src.train import choose_best_model, ModelResult, BestModelResult
=======
    from electricity_forecast.train import choose_best_model, ModelResult, BestModelResult
>>>>>>> f8f64ecb666768cf0d9ec227bc7ae180a7defcb5

    train_df, _ = train_and_test_df

    # Fake existing model
    existing_model = {
        "model": MagicMock(),
        "cv_rmse": 0.3,
        "model_type": "lr",
        "version": "4"
    }
    mock_compare.return_value = (True, existing_model)
    mock_evaluate.return_value = [ModelResult(model_type="lr", cv_rmse=0.5)]
    mock_get_best.return_value = ModelResult(model_type="lr", cv_rmse=0.5)

    # Fake MLflow run context
    mock_run = MagicMock()
    mock_run.__enter__.return_value = MagicMock(info=MagicMock(run_id="fake_run_id"))
    mock_start_run.return_value = mock_run

    # Run
    result = choose_best_model(train_df=train_df, config=config)

    # Result checks
    assert isinstance(result, BestModelResult)
    assert result.model_name == "lr"
    assert result.cv_rmse == 0.3
    assert result.is_from_mlflow
    assert result.production_version == "4"

    # Call assertions
    mock_setup_mlflow.assert_called_once_with(config)
    assert mock_start_run.call_count == 2
    mock_evaluate.assert_called_once()
    mock_get_best.assert_called_once_with(mock_evaluate.return_value)
    mock_compare.assert_called_once_with(mock_get_best.return_value, config)


<<<<<<< HEAD
@patch("scripts.train.mlflow.log_metric")
@patch("scripts.train.mlflow.log_param")
@patch("scripts.train.mlflow.log_params")
@patch("scripts.train.mlflow.set_tag")
@patch("scripts.train.mlflow.start_run")
@patch("scripts.train.setup_mlflow_experiment")
@patch("scripts.train.evaluate_candidate_models")
@patch("scripts.train.get_best_candidate_model")
@patch("scripts.train.compare_with_production_model")
@patch("scripts.train.retrain_full_model")
@patch("scripts.train.log_and_register_final_model")
=======
@patch("electricity_forecast.train.mlflow.log_metric")
@patch("electricity_forecast.train.mlflow.log_param")
@patch("electricity_forecast.train.mlflow.log_params")
@patch("electricity_forecast.train.mlflow.set_tag")
@patch("electricity_forecast.train.mlflow.start_run")
@patch("electricity_forecast.train.setup_mlflow_experiment")
@patch("electricity_forecast.train.evaluate_candidate_models")
@patch("electricity_forecast.train.get_best_candidate_model")
@patch("electricity_forecast.train.compare_with_production_model")
@patch("electricity_forecast.train.retrain_full_model")
@patch("electricity_forecast.train.log_and_register_final_model")
>>>>>>> f8f64ecb666768cf0d9ec227bc7ae180a7defcb5
def test_choose_best_model_trains_new(
    mock_log_and_register,
    mock_retrain_full,
    mock_compare,
    mock_get_best,
    mock_evaluate,
    mock_setup_mlflow,
    mock_start_run,
    mock_set_tag,
    mock_log_params,
    mock_log_param,
    mock_log_metric,
    config,
    train_and_test_df
):
    """Test that choose_best_model trains and registers a new model when it's better"""
<<<<<<< HEAD
    from src.train import choose_best_model, ModelResult, BestModelResult
=======
    from electricity_forecast.train import choose_best_model, ModelResult, BestModelResult
>>>>>>> f8f64ecb666768cf0d9ec227bc7ae180a7defcb5
    train_df, _ = train_and_test_df
    X_train = train_df.drop(columns=[config['data']['target_column']])
    y_train = train_df[config['data']['target_column']]
    
    # Mock new model is better than existing
    mock_compare.return_value = (False, None)  # Don't use existing model
    
    # Mock candidate model evaluation
    best_candidate = ModelResult(
        model_type="rf", 
        cv_rmse=0.2,
        model_config={"n_estimators": 100},
        best_params={"max_depth": 5}
    )
    mock_evaluate.return_value = [best_candidate, ModelResult(model_type="lr", cv_rmse=0.4)]
    mock_get_best.return_value = best_candidate
    
    # Mock retraining and model registration
    mock_final_model = MagicMock()
    mock_retrain_full.return_value = mock_final_model
    mock_log_and_register.return_value = "1"  # New model version
    
    # Fake MLflow run context
    mock_run = MagicMock()
    mock_run.__enter__.return_value = MagicMock(info=MagicMock(run_id="fake_run_id"))
    mock_start_run.return_value = mock_run
    
    # Run
    result = choose_best_model(train_df=train_df, config=config)
    
    # Result checks
    assert isinstance(result, BestModelResult)
    assert result.model_name == "rf"
    assert result.cv_rmse == 0.2
    assert not result.is_from_mlflow  # New model, not from MLflow
    assert result.production_version == "1"
    assert result.full_pipeline == mock_final_model
    
    # Call assertions
    mock_setup_mlflow.assert_called_once_with(config)
    assert mock_start_run.call_count == len(config['models']) + 1
    mock_evaluate.assert_called_once()
    mock_get_best.assert_called_once_with(mock_evaluate.return_value)
    mock_compare.assert_called_once_with(best_candidate, config)
    
    # Verify retraining was called with correct parameters
    mock_retrain_full.assert_called_once()
    
    # Verify model registration was called
    mock_log_and_register.assert_called_once_with(
        mock_final_model, "rf", 0.2, config
    )

