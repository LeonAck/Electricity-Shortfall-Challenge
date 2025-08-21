import pytest
import tempfile
import shutil
import os
from pathlib import Path
from unittest.mock import patch, MagicMock
import pandas as pd
from pydantic import ValidationError
from hydra.errors import MissingConfigException

from electricity_forecast.data_loading import load_data, test_training_data
from electricity_forecast.config_and_logging import load_config_hydra
from electricity_forecast.train import choose_best_model
from electricity_forecast.main import main

# Get the project root directory  
project_root = Path(__file__).parent.parent.parent
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


@pytest.fixture
def mock_best_existing_model():
    """
    Fixture that returns a mock best existing model with perfect RMSE
    This ensures the existing model is always chosen without MLflow lookup
    """
    return {
        'model': None,
        'model_type': 'Ridge', 
        'cv_rmse': float(0),  # Perfect score ensures this model is always "best"
        'run_id': 'test_run_id',
        'version': 'test_version'
    }

@pytest.fixture
def sample_data():
    """Create sample data for testing"""
    # Create a small dataset for testing
    return pd.DataFrame({
    'Valencia_wind_deg': ['level_10', 'level_20', 'level_20', 'level_10', 'level_20', 'level_20', 'level_10', 'level_20', 'level_20'],
    'Seville_pressure': ['sp1010', 'sp1020', 'sp1030', 'sp1010', 'sp1020', 'sp1030','sp1010', 'sp1020', 'sp1030'],
    'temperature': [15.0, 16.5, 134, 15.0, 16.5, 14, 15.0, 16.5, 14],
    'time': ['2023-01-01 00:00:00', '2023-01-01 03:00:00', '2023-01-01 06:00:00', '2023-01-01 09:00:00', '2023-01-01 12:00:00', '2023-01-01 15:00:00', '2023-01-01 18:00:00', '2023-01-01 21:00:00', '2023-01-02 00:00:00'],
    'load_shortfall_3h': [1, 0, 1, 1, 0, 1, 1, 0, 1]
})


class TestFullPipeline:
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test outputs"""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
  
    @patch('electricity_forecast.train.mlflow.start_run')
    @patch('electricity_forecast.main.load_data')
    @patch('electricity_forecast.train.get_best_existing_model')
    @patch('electricity_forecast.main.test_training_data')
    def test_pipeline_smoke_test_with_mocked_data(self, mock_test_training_data, mock_get_best_existing, mock_load_data, mock_start_run, sample_data, mock_best_existing_model):
        """
        Smoke test - just verify the pipeline runs without crashing with mocked data
        """

        mock_run = MagicMock()
        mock_run.__enter__.return_value = MagicMock(info=MagicMock(run_id="test_run_id"))
        mock_start_run.return_value = mock_run
    
        mock_load_data.return_value = (sample_data, sample_data, sample_data)
        mock_test_training_data.return_value = True

        mock_get_best_existing.return_value = mock_best_existing_model
        
        # This should not raise any exceptions
        result = main(config_name="config_test.yaml")
        
        # Basic assertions
        mock_load_data.assert_called_once()


    @patch('electricity_forecast.train.mlflow.start_run')
    @patch('electricity_forecast.main.load_data')
    def test_pipeline_with_real_config_and_mocked_data(self, mock_load_data, mock_start_run, config, sample_data):
        """
        Test with real config but mocked data
        """
        
        mock_load_data.return_value = (sample_data, sample_data, sample_data)

        # Mock MLflow run context
        mock_run = MagicMock()
        mock_run.__enter__.return_value = MagicMock(info=MagicMock(run_id="test_run_id"))
        mock_start_run.return_value = mock_run
        
        # Use real config loading with mocked data
        result = main(config_name="config_cv.yaml")
        
        # Add assertions about the result
        mock_load_data.assert_called_once()

    @patch('electricity_forecast.train.mlflow.start_run')
    def test_pipeline_with_real_data_loading(self, mock_start_run, test_config, train_and_test_df):
        """
        Test with real config and real data loading using train_and_test_df fixture
        """
        train_df, test_df = train_and_test_df
        
        # Verify we got real data
        assert not train_df.empty
        assert not test_df.empty
        
        # Test that the data works with our pipeline components
        test_training_data(train_df)

        # Mock MLflow run context
        mock_run = MagicMock()
        mock_run.__enter__.return_value = MagicMock(info=MagicMock(run_id="test_run_id"))
        mock_start_run.return_value = mock_run
        
        # Test model training with real data
        result = choose_best_model(train_df, test_config)
        assert result is not None
        assert hasattr(result, 'model_name')
        assert hasattr(result, 'cv_rmse')


    @patch('electricity_forecast.train.mlflow.start_run')
    @patch('electricity_forecast.train.mlflow.log_metric')
    @patch('electricity_forecast.train.mlflow.log_param')
    def test_pipeline_end_to_end_with_real_config(self, mock_log_param, mock_log_metric, 
                                                 mock_start_run, config, sample_data):
        """
        End-to-end test with real config and MLflow mocked
        """
        # Mock MLflow run context
        mock_run = MagicMock()
        mock_run.__enter__.return_value = MagicMock(info=MagicMock(run_id="test_run_id"))
        mock_start_run.return_value = mock_run
        
        with patch('electricity_forecast.main.load_data') as mock_load_data:
            mock_load_data.return_value = (sample_data, sample_data, sample_data)
            
            # Run pipeline with real config
            train_df, _, _ = load_data(config)
            test_training_data(train_df)
            
            result = choose_best_model(train_df, config)
            
            # Assertions
            assert result is not None
            assert hasattr(result, 'model_name')
            assert hasattr(result, 'cv_rmse')
            
            # Verify MLflow interactions occurred
            assert mock_start_run.called
            assert mock_log_metric.called or mock_log_param.called

    @patch('electricity_forecast.train.mlflow.start_run')
    @patch('electricity_forecast.train.mlflow.log_metric') 
    @patch('electricity_forecast.train.mlflow.log_param')
    def test_pipeline_end_to_end_with_real_data(self, mock_log_param, mock_log_metric,
                                               mock_start_run, test_config, train_and_test_df):
        """
        End-to-end test with real data and real config, MLflow mocked
        """
        # Mock MLflow run context
        mock_run = MagicMock()
        mock_run.__enter__.return_value = MagicMock(info=MagicMock(run_id="test_run_id"))
        mock_start_run.return_value = mock_run
        
        train_df, _ = train_and_test_df
        
        # Run pipeline components
        test_training_data(train_df)
        result = choose_best_model(train_df, test_config)
        
        # Assertions
        assert result is not None
        assert hasattr(result, 'model_name')
        assert hasattr(result, 'cv_rmse')
        
        # Verify MLflow interactions occurred
        assert mock_start_run.called


class TestPipelineComponents:
    """Test individual components work together"""
    
    def test_data_loading_and_training_integration_with_real_config(self, config, sample_data):
        """Test that data loading output works with training input using real config"""
        
        with patch('electricity_forecast.data_loading.load_data') as mock_load_data:
            mock_load_data.return_value = (sample_data, sample_data, sample_data)
            
            # Load data with real config
            train_df, val_df, test_df = load_data(config)
            
            # Test that training can accept this data
            test_training_data(train_df)  # Should not raise
            
            # Test that choose_best_model can accept this data
            with patch('electricity_forecast.train.mlflow.start_run'):
                result = choose_best_model(train_df, config)
                assert result is not None
                
    @patch('electricity_forecast.train.mlflow.start_run')
    def test_data_loading_and_training_integration_with_real_data(self, mock_start_run, test_config, train_and_test_df):
        """Test components work together with real data loading"""
        # Mock MLflow run context
        mock_run = MagicMock()
        mock_run.__enter__.return_value = MagicMock(info=MagicMock(run_id="test_run_id"))
        mock_start_run.return_value = mock_run

        train_df, test_df = train_and_test_df
        
        # Test that training can accept this real data
        test_training_data(train_df)  # Should not raise
        
        # Test that choose_best_model can accept this real data
        
        result = choose_best_model(train_df, test_config)
        assert result is not None

# Error Handling Tests
class TestPipelineErrorHandling:
    
    def test_pipeline_handles_bad_data(self):
        """Test pipeline handles corrupted/bad data gracefully"""
        # Create problematic data
        bad_data = pd.DataFrame({
            'feature1': [None, None, None],
            'target': [1, 0, None]
        })
        
        with pytest.raises(ValidationError):
            test_training_data(bad_data)

    def test_pipeline_handles_missing_config(self):
        """Test pipeline handles missing configuration"""
        with pytest.raises(MissingConfigException):
            load_config_hydra(config_name="nonexistent.yaml", config_path=str(config_path))

    def test_invalid_config_path(self):
        """Test handling of invalid config path"""
        with pytest.raises(MissingConfigException):
            load_config_hydra(config_name="config_test.yaml", config_path="/invalid/path")

    def test_real_data_validation(self, train_and_test_df):
        """Test that real data passes validation"""
        train_df, test_df = train_and_test_df
        
        # These should not raise exceptions
        test_training_data(train_df)
        
        # Basic data quality checks
        assert not train_df.empty
        assert not test_df.empty
        assert train_df.shape[0] > 0
        assert train_df.shape[1] > 0


if __name__ == "__main__":
    # Run specific test
    pytest.main([__file__ + "::TestFullPipeline::test_pipeline_smoke_test", "-v"])