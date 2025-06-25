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

    print("Data preprocessen...")
    
    # Create preprocessing pipeline with your preferred imputation method
    pipeline = create_preprocessing_pipeline(
        imputer     = get_imputer(config),
        freq        = config['preprocessing']['freq'],
        fill_method = config['preprocessing']['fill_method'],
        add_time_dummies = config['preprocessing']['add_time_dummies']                                             
    )

    # Fit the pipeline on training data
    pipeline.fit(train_df)

    # Transform both training and test data
    train_processed = pipeline.transform(train_df)
    test_processed = pipeline.transform(test_df)
    
    # Load models
    model_cfgs = config['models']
    models_to_try = {
    mc['type']: get_model(mc['type'], mc['params'])
    for mc in model_cfgs
    }

    # Model selection
    best_rmse, best_model, best_model_name = choose_best_model(output_dir, train_processed, models_to_try)

    metrics = {"rmse_validation": best_rmse, "model": best_model_name}
    save_run_metadata(output_dir, config, metrics)

    # Log to MLflow
    print(config.get("models", {}))
          
    log_to_mlflow(config, output_dir, run_id, best_model_name, best_model, metrics, parameters=config.get("models", {}))
    
    if config['run']['submit']:
        # Train on full training set and predict on test set
        test_predictions = train_full_model_predict_test_set(best_model, train_processed, test_processed, target_column=target_column)

        # Output
        submission_df = pd.DataFrame({
            'time': test_processed.index,
            'load_shortfall_3h': test_predictions
        })
        submission_df.to_csv('sample_submission.csv', index=False)
        print("\nVoorspellingen opgeslagen in 'sample_submission.csv'")

if __name__ == "__main__":
    config_path = 'Configs/deep1.yaml'
    main(config_path=config_path)