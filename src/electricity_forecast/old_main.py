from src.data_loading import load_data, test_training_data
from src.config_and_logging import generate_run_id, save_run_metadata, create_output_dir, log_to_mlflow, save_model_and_pipeline, run_with_config, load_config_hydra
from src.model_pipeline import choose_best_model
from src.train import choose_best_model

import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path



# Get the project root directory
project_root = Path(__file__).parent.parent
config_path = project_root / "configs"


def main(config_name="config.yaml"):
    config = load_config_hydra(config_name=config_name, config_path=str(config_path))
    
    run_name = config['run']['run_name']

    run_id = generate_run_id(config)
    output_dir = create_output_dir(run_name, run_id, config)

    print("Run name:", run_name)
    print("Run ID:", run_id)
    print("Load data...")
    train_df, _, _ = load_data(config)

    test_training_data(train_df)

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

    if config['logging']['mlflow_enabled']:
        # Log to MLflow
        log_to_mlflow(config, output_dir, run_id, best_model_results['model_name'], best_model_results["model_object"], metrics, parameters=config.get("models", {}))

    if config['output']['save_model']:
        # Create a folder for saved models and save the best model and preprocessing pipeline
        save_model_and_pipeline(best_model_results["pipeline"], best_model_results['model_object'], config)


if __name__ == "__main__":
    main()
