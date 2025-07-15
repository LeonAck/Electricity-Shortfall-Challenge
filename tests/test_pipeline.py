import os
import pytest
import pandas as pd

from data_loading import load_data
from config_and_logging import load_config
from model_pipeline import choose_best_model
from models import get_model
from preprocessing import get_imputer, create_preprocessing_pipeline

@pytest.fixture
def config():
    # Adjust path if needed
    return load_config('Configs/shallow2_scaling_timeseriessplit.yaml')

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

def test_preprocessing_pipeline_shapes(config, train_and_test_df):
    train_df, _ = train_and_test_df
    X = train_df.drop(columns=[config['data']['target_column']])
    imputer = get_imputer(config)
    pipeline = create_preprocessing_pipeline(
        imputer,
        freq=config['preprocessing']['freq'],
        fill_method=config['preprocessing']['fill_method'],
        add_time_dummies=config['preprocessing']['add_time_dummies'],
        scaling=True
    )
    X_transformed = pipeline.fit_transform(X)
    assert X_transformed.shape[0] == X.shape[0]

def test_choose_best_model_logic(config, train_and_test_df):
    train_df, _ = train_and_test_df
    y_train = train_df[config['data']['target_column']]
    X_train = train_df.drop(columns=[config['data']['target_column']])

    imputer = get_imputer(config)
    pipeline = create_preprocessing_pipeline(
        imputer,
        freq=config['preprocessing']['freq'],
        fill_method=config['preprocessing']['fill_method'],
        add_time_dummies=config['preprocessing']['add_time_dummies'],
        scaling=True
    )
    X_train_transformed = pipeline.fit_transform(X_train)

    models_to_try = {
        'ridge': {
            'model': get_model('ridge', {}),
            'X_train': X_train_transformed,
            'X_test': X_train_transformed.copy()  # dummy for test
        }
    }

    rmse, model, model_name, _, _ = choose_best_model(
        output_dir='.', 
        y=y_train, 
        models=models_to_try, 
        train_val_split=config['preprocessing']['train_val_split']
    )

    assert model is not None
    assert isinstance(rmse, float)
    assert rmse > 0
