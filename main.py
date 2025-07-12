from data_loading import load_data
from config_and_logging import load_config, generate_run_id, save_run_metadata, create_output_dir, log_to_mlflow
from model_pipeline import choose_best_model, train_full_model_predict_test_set
from models import get_model
from preprocessing import get_imputer, create_preprocessing_pipeline

import pandas as pd
import os

def main(config_path):
    config = load_config(config_path=os.path.join(os.getcwd(), config_path))
    run_name = config['run']['run_name']
    run_id = generate_run_id(config)
    output_dir = create_output_dir(run_name, run_id)

    target_column = config['data']['target_column']
    print("Run name:", run_name)
    print("Run ID:", run_id)
    print("Data laden...")
    train_df, test_df, sample_submission = load_data(config)

    print("Pipelines aanmaken...")

    # Shared preprocessing config
    imputer = get_imputer(config)
    freq = config['preprocessing']['freq']
    fill_method = config['preprocessing']['fill_method']
    add_time_dummies = config['preprocessing']['add_time_dummies']

    # Split X and y sets
    y_train = train_df[target_column]
    X_train = train_df.drop(columns=[target_column])

    # Pipelines
    pipeline_scaled = create_preprocessing_pipeline(imputer, freq, fill_method, add_time_dummies, scaling=True)
    pipeline_no_scaling = create_preprocessing_pipeline(imputer, freq, fill_method, add_time_dummies, scaling=False)

    # Fit both pipelines on training data
    X_train_scaled = pipeline_scaled.fit_transform(X_train)
    X_train_no_scaling = pipeline_no_scaling.fit_transform(X_train)

    # Transform test set as well (will be needed later)
    test_scaled = pipeline_scaled.transform(test_df)
    test_no_scaling = pipeline_no_scaling.transform(test_df)

    # Load models
    model_cfgs = config['models']
    models_to_try = {}

    for mc in model_cfgs:
        model_name = mc['type']
        scaling_needed = mc.get('scaling', False)

        if scaling_needed:
            X_train_transformed = X_train_scaled
            X_test_transformed = test_scaled
        else:
            X_train_transformed = X_train_no_scaling
            X_test_transformed = test_no_scaling

        model = get_model(model_name, mc['params'])

        models_to_try[model_name] = {
            'model': model,
            'X_train': X_train_transformed.copy(),
            'X_test': X_test_transformed.copy()
        }

    # Model selection
    best_rmse, best_model, best_model_name, best_X_train, best_X_test = choose_best_model(
        output_dir, 
        y_train, 
        models_to_try,
        config['preprocessing']['train_val_split']
    )

    metrics = {"rmse_validation": best_rmse, "model": best_model_name}

    save_run_metadata(output_dir, config, metrics)

    # Log to MLflow
    log_to_mlflow(config, output_dir, run_id, best_model_name, best_model, metrics, parameters=config.get("models", {}))

    if config['run']['submit']:
        # Train on full set and predict on test set
        test_predictions = train_full_model_predict_test_set(
            best_model, 
            best_X_train, 
            best_X_test, 
            y_train
        )

        submission_df = pd.DataFrame({
            'time': test_df.index,  # or test_df['time'] if that's your column
            'load_shortfall_3h': test_predictions
        })
        submission_df.to_csv('sample_submission.csv', index=False)
        print("\nVoorspellingen opgeslagen in 'sample_submission.csv'")

if __name__ == "__main__":
    config_path = 'Configs/shallow2_scaling_timeseriessplit.yaml'
    main(config_path=config_path)
