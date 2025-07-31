import os
import sys
import pytest
import joblib
import hydra
from omegaconf import DictConfig
from pathlib import Path

from scripts.data_loading import load_data
from scripts.config_and_logging import load_config, load_config_hydra
from scripts.model_pipeline import choose_best_model
from scripts.inference import load_models, predict_batch



# Get the project root directory
project_root = Path(__file__).parent.parent
config_path = project_root / "configs"


@pytest.fixture
def config(config_name="config.yaml"):
    # Get the project root directory
    project_root = Path(__file__).parent.parent
    config_path = project_root / "configs"

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config directory not found at {config_path}")

    # Adjust path if needed
    return load_config_hydra(config_name=config_name, config_path=str(config_path))

@pytest.fixture
def test_config(config_name="config_test.yaml"):
    # Get the project root directory
    project_root = Path(__file__).parent.parent
    config_path = project_root / "configs"

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config directory not found at {config_path}")

    # Adjust path if needed
    return load_config_hydra(config_name=config_name, config_path=str(config_path))


@pytest.fixture
def train_and_test_df(config):
    train_df, test_df, _ = load_data(config)
    return train_df, test_df

def test_load_config_and_data(config, train_and_test_df):
    train_df, _ = train_and_test_df
    assert 'run' in config
    assert 'models' in config
    assert not train_df.empty
    assert config['data']['target_column'] in train_df.columns

def test_choose_best_model_logic(config, train_and_test_df):
    """
    Test the logic of choosing the best model based on RMSE.
    """
    train_df, _ = train_and_test_df

    best_model_results = choose_best_model(
        output_dir='.', 
        train_df=train_df,
        config=config
    )

    assert best_model_results["model_object"] is not None
    assert best_model_results["pipeline"] is not None
    assert isinstance(best_model_results['rmse'], float)
    assert best_model_results['rmse'] > 0


def test_ARIMA_predict(train_and_test_df, test_config):
    """
    Test the ARIMA model prediction functionality.
    """

    train_df, test_df = train_and_test_df

    best_model_results = choose_best_model(
        output_dir='.', 
        train_df=train_df,
        config=test_config
    )

    # Create a folder for saved models
    joblib.dump(best_model_results["pipeline"], "tests/preprocessing_pipeline.pkl")
    joblib.dump(best_model_results['model_object'], "tests/best_model.pkl")
    
    model, pipeline = load_models(folder='tests')
    if model and pipeline:
        preds = predict_batch(test_df, model, pipeline)
        assert preds.shape[0] == test_df.shape[0]
        assert not preds.isnull().any()

