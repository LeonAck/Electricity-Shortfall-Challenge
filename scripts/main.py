from scripts.data_loading import load_data
from scripts.config_and_logging import load_config, generate_run_id, save_run_metadata, create_output_dir, log_to_mlflow, save_model_and_pipeline
from scripts.model_pipeline import choose_best_model
from scripts.inference import predict_batch

import pandas as pd
import os
import joblib

def main(config_path):
    config = load_config(config_path=os.path.join(os.getcwd(), config_path))
    run_name = config['run']['run_name']
    run_id = generate_run_id(config)
    output_dir = create_output_dir(run_name, run_id)

    print("Run name:", run_name)
    print("Run ID:", run_id)
    print("Load data...")
    train_df, test_df, _ = load_data(config)

    print("Choose best models on training and validation set. Retrain on on full training set...")

    # Model selection
    best_model_results = choose_best_model(
        output_dir,
        train_df, 
        config
    )

    metrics = {"rmse_validation": best_model_results['rmse'], "model": best_model_results['model_name']}

    print("Log and save results...")

    # Save config, metrics and model type
    save_run_metadata(output_dir, config, metrics)

    if config['logging']['mlflow_log']:
        # Log to MLflow
        log_to_mlflow(config, output_dir, run_id, best_model_results['model_name'], best_model_results["model_object"], metrics, parameters=config.get("models", {}))

    if config['output']['save_model']:
        # Create a folder for saved models and save the best model and preprocessing pipeline
        save_model_and_pipeline(best_model_results["pipeline"], best_model_results['model_object'], config)

    if config['run']['submit']:
        # Train on full set and predict on test set
        test_predictions = predict_batch(test_df, best_model_results['model_object'], best_model_results["pipeline"])

        submission_df = pd.DataFrame(
            test_predictions
        )
        submission_df.to_csv(f"{config['output']['output_folder']}/{config['output']['sample_submission_filename']}", index=False)
        print(f"\nPredictions saved in '{config['output']['sample_submission_filename']}'")


if __name__ == "__main__":
    config_path = 'configs/shallow4.yaml'
    main(config_path=config_path)
