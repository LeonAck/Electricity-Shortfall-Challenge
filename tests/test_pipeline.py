import os
import sys
import pytest
import joblib

# Debug: Let's see what's actually available
print(f"=== FULL DEBUG INFO ===")
print(f"Current file: {__file__}")
print(f"Absolute path: {os.path.abspath(__file__)}")
print(f"Current working directory: {os.getcwd()}")

# Check multiple directories
directories_to_check = [
    os.path.dirname(os.path.abspath(__file__)),  # tests directory
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),  # parent
    os.getcwd(),  # current working dir
    "/home/runner/work/Electricity-Shortfall-Challenge",  # potential root
    "/home/runner/work/Electricity-Shortfall-Challenge/Electricity-Shortfall-Challenge",  # double nested
]

for i, directory in enumerate(directories_to_check):
    print(f"\n--- Directory {i+1}: {directory} ---")
    if os.path.exists(directory):
        try:
            contents = os.listdir(directory)
            print(f"Contents: {contents}")
            if 'configs' in contents:
                print(f"✓ FOUND configs directory here!")
            else:
                print("✗ No configs directory here")
        except PermissionError:
            print("Permission denied")
    else:
        print("Directory does not exist")

print(f"========================")

# Temporarily use a simple fallback to let the script continue
project_root = os.getcwd()  # Just use current directory for now
sys.path.insert(0, project_root)

# Keep the imports but don't run tests yet
try:
    from scripts.data_loading import load_data
    print("✓ Successfully imported scripts.data_loading")
except ImportError as e:
    print(f"✗ Failed to import scripts.data_loading: {e}")

# Simple test to see if this works
def test_debug():
    assert True  # This will always pass, just to see the debug output

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

