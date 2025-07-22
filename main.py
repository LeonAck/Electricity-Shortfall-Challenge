from data_loading import load_data
from config_and_logging import load_config, generate_run_id, save_run_metadata, create_output_dir, log_to_mlflow
from model_pipeline import choose_best_model
from inference import predict_batch

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
    print("Data laden...")
    train_df, test_df, _ = load_data(config)

    print("Choose best models...")

    # Model selection
    best_model_results = choose_best_model(
        output_dir,
        train_df, 
        config,
        config['preprocessing']['train_val_split']
    )

    metrics = {"rmse_validation": best_model_results['rmse'], "model": best_model_results['model_name']}

    save_run_metadata(output_dir, config, metrics)

    # Log to MLflow
    log_to_mlflow(config, output_dir, run_id, best_model_results['model_name'], best_model_results["model_object"], metrics, parameters=config.get("models", {}))

    # Create a folder for saved models
    os.makedirs('saved_models', exist_ok=True)
    joblib.dump(best_model_results["pipeline"], "saved_models/preprocessing_pipeline.pkl")
    joblib.dump(best_model_results['model_object'], "saved_models/best_model.pkl")

    if config['run']['submit']:
        # Train on full set and predict on test set
        test_predictions = predict_batch(test_df, best_model_results['model_object'], best_model_results["pipeline"])

        submission_df = pd.DataFrame(
            test_predictions
        )
        submission_df.to_csv('Output/sample_submission.csv', index=False)
        print("\nVoorspellingen opgeslagen in 'sample_submission.csv'")

if __name__ == "__main__":
    config_path = 'Configs/shallow4.yaml'
    main(config_path=config_path)
