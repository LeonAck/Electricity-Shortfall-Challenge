from electricity_forecast.data_loading import load_data, test_training_data
from electricity_forecast.config_and_logging import load_config_hydra, save_model_and_pipeline
from electricity_forecast.train import choose_best_model

from pathlib import Path
import logging

# Basic logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Get the project root directory
project_root = Path(__file__).parent.parent.parent
config_path = project_root / "configs"


def main(config_name="config_test.yaml"):
    config = load_config_hydra(config_name=config_name, config_path=str(config_path))
    print(config.__class__)
    
    print("Load data...")
    train_df, _, _ = load_data(config)

    test_training_data(train_df)

    print("Choose best models on training and validation set. Retrain on on full training set...")

    # Model selection
    best_model_results = choose_best_model(
        train_df, 
        config
    )

    print(best_model_results)
    save_model_and_pipeline(best_model_results.full_pipeline, config)

if __name__ == "__main__":
    main()

