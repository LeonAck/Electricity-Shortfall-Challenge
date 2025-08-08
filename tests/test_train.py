import os
import sys
import pytest
import joblib
import hydra
from omegaconf import DictConfig
from pathlib import Path
from unittest.mock import patch, MagicMock

from scripts.data_loading import load_data
from scripts.config_and_logging import load_config_hydra
from scripts.train import (
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

@patch("scripts.train.get_pipeline_for_model")
@patch("scripts.train.get_split_type")
def test_train_with_cross_validation(config, train_and_test_df):
    train_df, _ = train_and_test_df
    X_train = train_df.drop(columns=[config['data']['target_column']])
    y_train = train_df[config['data']['target_column']]

    model_config = config['models'][0]  # First model in config

    with patch("scripts.models.get_model") as mock_get_model, \
         patch("scripts.preprocessing.get_pipeline_for_model") as mock_get_pipeline:

        # Mock model
        mock_model = MagicMock()
        mock_model.fit = MagicMock()
        mock_model.predict = MagicMock(return_value=y_train[:len(y_train)])  # dummy pred
        mock_get_model.return_value = mock_model

        # Mock pipeline
        mock_pipeline = MagicMock()
        mock_pipeline.fit_transform.return_value = X_train  # pass-through
        mock_get_pipeline.return_value = mock_pipeline

        cv_rmse = train_with_cross_validation(model_config, config, X_train, y_train)

        assert isinstance(cv_rmse, float)
        assert cv_rmse > 0
        mock_get_model.assert_called()
        mock_get_pipeline.assert_called()


@patch("scripts.train.get_pipeline_for_model")
@patch("scripts.train.get_split_type")
@patch("scripts.train.GridSearchCV")
def test_train_with_hyperparameter_tuning_no_params(config, train_and_test_df):
    train_df, _ = train_and_test_df
    X_train = train_df.drop(columns=[config['data']['target_column']])
    y_train = train_df[config['data']['target_column']]

    # Model without tuning params
    model_config = {
        "type": "linear_regression",
        "params": {},
        "tuning_params": {}
    }

    with patch("scripts.train.train_with_cross_validation") as mock_cv:
        mock_cv.return_value = 0.123
        cv_rmse, best_params = train_with_hyperparameter_tuning(model_config, config, X_train, y_train)
        assert cv_rmse == 0.123
        assert best_params is None


@patch("scripts.train.get_pipeline_for_model")
@patch("scripts.train.get_split_type")
@patch("scripts.train.GridSearchCV")
def test_train_with_hyperparameter_tuning_with_params(mock_grid_search, config, train_and_test_df):
    train_df, _ = train_and_test_df
    X_train = train_df.drop(columns=[config['data']['target_column']])
    y_train = train_df[config['data']['target_column']]

    # Pick a model with tuning params (e.g., second model)
    model_config = config['models'][1] if len(config['models']) > 1 else {
        "type": "random_forest",
        "params": {"n_estimators": 10},
        "tuning_params": {"n_estimators": [10, 20]}
    }

    with patch("scripts.models.get_model") as mock_get_model, \
         patch("scripts.preprocessing.get_pipeline_for_model") as mock_get_pipeline:

        mock_model = MagicMock()
        mock_get_model.return_value = mock_model

        mock_pipeline = MagicMock()
        mock_pipeline.fit_transform.return_value = X_train.values
        mock_get_pipeline.return_value = mock_pipeline

        # Mock GridSearchCV result
        mock_search = MagicMock()
        mock_search.best_score_ = -0.45
        mock_search.best_params_ = {"n_estimators": 20}
        mock_grid_search.return_value = mock_search

        cv_rmse, best_params = train_with_hyperparameter_tuning(model_config, config, X_train, y_train)

        assert cv_rmse == 0.45
        assert best_params == {"n_estimators": 20}
        mock_grid_search.assert_called()

@patch("scripts.train.get_pipeline_for_model")
@patch("scripts.models.get_model")
def test_retrain_full_model(config, train_and_test_df):
    train_df, _ = train_and_test_df
    X_train = train_df.drop(columns=[config['data']['target_column']])
    y_train = train_df[config['data']['target_column']]

    model_config = config['models'][0]

    with patch("scripts.models.get_model") as mock_get_model, \
         patch("scripts.preprocessing.get_pipeline_for_model") as mock_get_pipeline:

        mock_model = MagicMock()
        mock_get_model.return_value = mock_model

        mock_pipeline = MagicMock()
        mock_pipeline.fit_transform.return_value = X_train.values
        mock_get_pipeline.return_value = mock_pipeline

        final_pipeline = retrain_full_model(X_train, y_train, config, model_config)

        assert hasattr(final_pipeline, 'fit')
        assert hasattr(final_pipeline, 'predict')
        assert len(final_pipeline.named_steps) >= 2
        mock_model.fit.assert_called()


def test_evaluate_candidate_models(config, train_and_test_df):
    train_df, _ = train_and_test_df
    X_train = train_df.drop(columns=[config['data']['target_column']])
    y_train = train_df[config['data']['target_column']]

    # Temporarily disable tuning to simplify test
    with patch("scripts.train.train_with_cross_validation", return_value=0.5), \
         patch("scripts.train.train_with_hyperparameter_tuning", return_value=(0.45, {"alpha": 0.1})):

        # Temporarily override config tuning setting
        config_copy = config.copy()
        config_copy.model_selection.tuning = True

        results = evaluate_candidate_models(config_copy, X_train, y_train)

        assert len(results) == len(config_copy.models)
        for res in results:
            assert isinstance(res, ModelResult)
            assert isinstance(res.cv_rmse, float)
            assert res.cv_rmse > 0


def test_get_best_candidate_model():
    results = [
        ModelResult(model_type="A", cv_rmse=0.6),
        ModelResult(model_type="B", cv_rmse=0.4),
        ModelResult(model_type="C", cv_rmse=0.5),
    ]
    best = get_best_candidate_model(results)
    assert best.model_type == "B"
    assert best.cv_rmse == 0.4


@patch("scripts.train.get_best_existing_model")
def test_compare_with_production_model_better_existing(mock_get_existing, config, train_and_test_df):
    mock_get_existing.return_value = {"cv_rmse": 0.3}

    train_df, _ = train_and_test_df
    X_train = train_df.drop(columns=[config['data']['target_column']])
    y_train = train_df[config['data']['target_column']]

    candidate = ModelResult(
        model_type="new_model",
        cv_rmse=0.35,
        best_params={},
        model_config={}
    )

    use_existing, existing = compare_with_production_model(candidate, config)
    assert use_existing is True
    assert existing["cv_rmse"] == 0.3


@patch("scripts.train.get_best_existing_model")
def test_compare_with_production_model_better_new(mock_get_existing, config, train_and_test_df):
    mock_get_existing.return_value = {"cv_rmse": 0.5}

    candidate = ModelResult(
        model_type="new_model",
        cv_rmse=0.4,
        best_params={},
        model_config={}
    )

    use_existing, existing = compare_with_production_model(candidate, config)
    assert use_existing is False
    assert existing is None


@patch("scripts.train.mlflow.log_metric")
@patch("scripts.train.mlflow.set_tag")
@patch("scripts.train.mlflow.start_run")
@patch("scripts.train.setup_mlflow_experiment")
@patch("scripts.train.evaluate_candidate_models")
@patch("scripts.train.get_best_candidate_model")
@patch("scripts.train.compare_with_production_model")
def test_choose_best_model_use_existing(
    mock_compare,
    mock_get_best,
    mock_evaluate,
    mock_setup_mlflow,
    mock_start_run,
    mock_set_tag,
    mock_log_metric,
    config,
    train_and_test_df
):
    train_df, _ = train_and_test_df

    existing_model = {
        "model": MagicMock(),
        "cv_rmse": 0.3,
        "model_type": "lr",
        "version": "4"
    }
    mock_compare.return_value = (True, existing_model)

    mock_run = MagicMock()
    mock_run.__enter__.return_value = MagicMock(info=MagicMock(run_id="test_run_123"))
    mock_run.__exit__.return_value = None
    mock_start_run.return_value = mock_run

    mock_evaluate.return_value = [ModelResult(model_type="lr", cv_rmse=0.5)]
    mock_get_best.return_value = ModelResult(model_type="lr", cv_rmse=0.5)

    result = choose_best_model(output_dir=".", train_df=train_df, config=config)

    assert isinstance(result, BestModelResult)
    assert result.model_name == "lr"
    assert abs(result.cv_rmse - 0.3) < 1e-5
    assert result.is_from_mlflow is True
    assert result.production_version == "4"


@patch("mlflow.MlflowClient")
def test_get_best_existing_model_failure(mock_client_class, config):
    mock_client = MagicMock()
    mock_client.get_model_version_by_alias.side_effect = Exception("NotFound")
    mock_client_class.return_value = mock_client

    result = get_best_existing_model(config)
    assert result is None


# ========================================
# Integration Test: choose_best_model
# ========================================

@patch("scripts.train.mlflow.start_run")
@patch("scripts.train.setup_mlflow_experiment")
@patch("scripts.train.evaluate_candidate_models")
@patch("scripts.train.get_best_candidate_model")
@patch("scripts.train.compare_with_production_model")
@patch("scripts.train.retrain_full_model")
@patch("scripts.train.log_and_register_final_model")
def test_choose_best_model_new_model(
    mock_log_register,
    mock_retrain,
    mock_compare,
    mock_get_best,
    mock_evaluate,
    mock_setup_mlflow,
    config,
    train_and_test_df
):
    train_df, _ = train_and_test_df

    # Mock candidate models
    mock_evaluate.return_value = [
        ModelResult(model_type="lr", cv_rmse=0.5),
        ModelResult(model_type="rf", cv_rmse=0.45)
    ]
    mock_get_best.return_value = ModelResult(model_type="rf", cv_rmse=0.45, best_params={"n_estimators": 10})
    mock_compare.return_value = (False, None)  # Use new model

    mock_final_model = MagicMock()
    mock_retrain.return_value = mock_final_model
    mock_log_register.return_value = "5"
    print(config['data']['target_column'])
    result = choose_best_model(output_dir=".", train_df=train_df, config=config)

    assert isinstance(result, BestModelResult)
    assert result.model_name == "rf"
    assert abs(result.cv_rmse - 0.45) < 1e-5
    assert result.is_from_mlflow is False
    assert result.production_version == "5"


@patch("scripts.train.setup_mlflow_experiment")
@patch("scripts.train.evaluate_candidate_models")
@patch("scripts.train.get_best_candidate_model")
@patch("scripts.train.compare_with_production_model")
def test_choose_best_model_use_existing(
    mock_compare,
    mock_get_best,
    mock_evaluate,
    mock_setup_mlflow,
    config,
    train_and_test_df
):
    train_df, _ = train_and_test_df

    existing_model = {
        "model": MagicMock(),
        "cv_rmse": 0.3,
        "model_type": "lr",
        "version": "4"
    }
    mock_compare.return_value = (True, existing_model)

    result = choose_best_model(output_dir=".", train_df=train_df, config=config)

    assert isinstance(result, BestModelResult)
    assert result.model_name == "lr"
    assert abs(result.cv_rmse - 0.3) < 1e-5
    assert result.is_from_mlflow is True
    assert result.production_version == "4"

    # Still evaluates candidates (for logging, comparison)
    mock_evaluate.assert_called()
    mock_get_best.assert_called()