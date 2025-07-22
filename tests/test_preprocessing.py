import pytest
import pandas as pd
import numpy as np
import os
import sys

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.preprocessing import WeatherDataPreprocessor, create_preprocessing_pipeline, StandardTransformerWrapper, SimplifiedPatternImputer, TimeAwareKNNImputer
from scripts.data_loading import load_data
from scripts.config_and_logging import load_config
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline



@pytest.fixture
def sample_df():
    return pd.DataFrame({
        'Valencia_wind_deg': ['level_10', 'level_20', np.nan],
        'Seville_pressure': ['sp1010', 'sp1020', 'sp1030'],
        'temperature': [15.0, 16.5, np.nan],
        'time': ['2023-01-01 00:00:00', '2023-01-01 03:00:00', '2023-01-01 06:00:00']
    })

@pytest.fixture
def config():
    # Adjust path if needed
    return load_config('configs/test_config.yaml')

@pytest.fixture
def train_and_test_df(config):
    train_df, test_df, _ = load_data(config)
    return train_df, test_df

# ---------- Preprocessing pipeline  tests ----------

def test_pipeline_default_structure():
    """
    Test that the preprocessing pipeline has the expected structure.
    """
    pipeline = create_preprocessing_pipeline(imputer=SimpleImputer(strategy='mean'))
    assert isinstance(pipeline, Pipeline)
    step_names = [name for name, _ in pipeline.steps]
    assert step_names == ['preprocessor', 'imputer', 'to_numpy']


def test_pipeline_runs_on_dummy_data():
    """
    Test that the preprocessing pipeline can run on dummy data.
    """
    df = pd.DataFrame({
        'temperature': [np.nan, 20, 21, 19],
        'humidity': [30, np.nan, 40, 45],
        'time': pd.date_range("2023-01-01", periods=4, freq="3h")
    })

    imputer = TimeAwareKNNImputer()
    pipeline = StandardTransformerWrapper(create_preprocessing_pipeline(imputer=imputer))
    output = pipeline.fit_transform(df)
    assert isinstance(output, np.ndarray)
    assert output.shape[0] == df.shape[0]  # Check row count preserved
    assert np.isnan(np.sum(output[:, 0])) == 1 # Check NaNs are not handled


def test_pipeline_scaling_included():
    """
    Test that scaling is included in the pipeline when specified.
    """
    imputer = SimpleImputer(strategy='mean')
    pipeline = create_preprocessing_pipeline(imputer=imputer, scaling=True)
    step_names = [name for name, _ in pipeline.steps]
    assert 'scaler' in step_names


def test_pipeline_scaling_excluded():
    """
    Test that scaling is excluded from the pipeline when specified.
    """
    imputer = SimpleImputer(strategy='mean')
    pipeline = create_preprocessing_pipeline(imputer=imputer, scaling=False)
    step_names = [name for name, _ in pipeline.steps]
    assert 'scaler' not in step_names

# ---------- Preprocessing tests ----------

def test_pipeline_other_imputer():
    """
    Test that the pipeline can use a different imputer than TimeAwareKNNImputer.
    """
    df = pd.DataFrame({
        'temperature': [np.nan, 20, 21, 19],
        'humidity': [30, np.nan, 40, 45],
        'time': pd.date_range("2023-01-01", periods=4, freq="3h")
    })

    imputer = SimplifiedPatternImputer(column='temperature')
    pipeline = StandardTransformerWrapper(create_preprocessing_pipeline(imputer=imputer))
    result = pipeline.fit_transform(df)
    assert isinstance(result, np.ndarray)
    assert np.isnan(np.sum(result[:,0])) == 0 # Check no NaNs remain


def test_time_dummies_affect_output_shape():
    """
    Test that adding cyclical time features adds 6 columns in the output array
    """
    imputer = TimeAwareKNNImputer()

    df = pd.DataFrame({
        'temperature': [20, 21, 19, 22],
        'humidity': [30, 35, 40, 45],
        'time': pd.date_range("2023-01-01", periods=4, freq="h")
    })

    pipeline_no_dummies = StandardTransformerWrapper(create_preprocessing_pipeline(imputer=imputer, add_time_dummies=None))
    output1 = pipeline_no_dummies.fit_transform(df)
    
    pipeline_with_dummies = StandardTransformerWrapper(create_preprocessing_pipeline(imputer=imputer, add_time_dummies='cyclical'))
    output2 = pipeline_with_dummies.fit_transform(df)

    assert output2.shape[1] == output1.shape[1] + 6  # 6 cyclical features added


def test_categorical_columns(sample_df):
    
    """
    Test that categorical columns are changed to numerical
    """
    preprocessor = WeatherDataPreprocessor()
    result = preprocessor.fit_transform(sample_df)

    # Check that converted columns exist
    assert result['Valencia_wind_deg'].dtype == 'float'
    assert result['Seville_pressure'].dtype == 'float'


def test_datetime_index(sample_df):
    preprocessor = WeatherDataPreprocessor()
    result = preprocessor.fit_transform(sample_df)
    assert isinstance(result.index, pd.DatetimeIndex)


def test_reindexing_and_dropping(sample_df):
    """
    Test that missing timestamps are filled and dropped correctly.
    """
    # Remove one time row to simulate missing timestamps
    sample_df_missing = sample_df.drop(index=1).reset_index(drop=True)
    
    preprocessor = WeatherDataPreprocessor()
    result = preprocessor.fit_transform(sample_df_missing)

    # Should include all timestamps from min to max in 3h steps
    expected_index = pd.date_range("2023-01-01 00:00:00", "2023-01-01 06:00:00", freq="3h")
    # Fully missing row is dropped
    assert set(result.index).issubset(set(expected_index))
    assert len(result) < len(expected_index)

# ---------- Test 4: Cyclical time features ----------
def test_cyclical_time_features(sample_df):
    preprocessor = WeatherDataPreprocessor(add_time_dummies="cyclical")
    result = preprocessor.fit_transform(sample_df)

    for col in ['hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'month_sin', 'month_cos']:
        assert col in result.columns

# ---------- Test 5: Drop Unnamed column ----------

def test_drops_unnamed_column(sample_df):
    df = sample_df.copy()
    df['Unnamed: 0'] = [1, 2, 3]
    preprocessor = WeatherDataPreprocessor()
    result = preprocessor.fit_transform(df)
    assert 'Unnamed: 0' not in result.columns


def test_no_nans(train_and_test_df):
    """
    Test that no NaNs remain after preprocessing
    """
    df, _ = train_and_test_df

    imputer = SimplifiedPatternImputer(column='Valencia_pressure')
    pipeline = StandardTransformerWrapper(create_preprocessing_pipeline(imputer=imputer))
    result = pipeline.fit_transform(df)

    assert isinstance(result, np.ndarray)
    assert np.isnan(np.sum(result[:,0])) == 0 # Check no NaNs remain

