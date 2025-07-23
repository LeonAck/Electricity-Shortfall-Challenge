import os
import sys

import joblib

# Find the actual project root by looking for the configs directory
def find_project_root():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Check current directory and parent directories
    check_dirs = [
        current_dir,  # tests directory
        os.path.dirname(current_dir),  # parent directory
        os.getcwd(),  # current working directory
    ]
    
    for directory in check_dirs:
        configs_path = os.path.join(directory, 'configs')
        if os.path.exists(configs_path):
            print(f"Found project root at: {directory}")
            return directory
    
    raise FileNotFoundError("Could not find project root with configs directory")

project_root = find_project_root()
sys.path.insert(0, project_root)

print(f"=== DEBUG INFO ===")
print(f"Project root: {project_root}")
print(f"Current working directory: {os.getcwd()}")
print(f"Contents of project root: {os.listdir(project_root)}")
print(f"Configs directory contents: {os.listdir(os.path.join(project_root, 'configs'))}")
print(f"==================")

from scripts.data_loading import load_data
from scripts.config_and_logging import load_config
from scripts.model_pipeline import choose_best_model
from scripts.inference import load_models, predict_batch


@pytest.fixture
def config():
    # Get the project root directory (where the tests are running from)
    config_path = os.path.join(project_root, 'configs/shallow4.yaml')

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")

    # Adjust path if needed
    return load_config(config_path)

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

def test_ARIMA_predict(train_and_test_df, config_path= "configs/test_config.yaml"):
    """
    Test the ARIMA model prediction functionality.
    """
    config = load_config(config_path)
    train_df, test_df = train_and_test_df

    best_model_results = choose_best_model(
        output_dir='.', 
        train_df=train_df,
        config=config
    )

    # Create a folder for saved models
    joblib.dump(best_model_results["pipeline"], "tests/preprocessing_pipeline.pkl")
    joblib.dump(best_model_results['model_object'], "tests/best_model.pkl")
    
    model, pipeline = load_models(folder='tests')
    if model and pipeline:
        preds = predict_batch(test_df, model, pipeline)
        assert preds.shape[0] == test_df.shape[0]
        assert not preds.isnull().any()

