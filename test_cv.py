from scripts.data_loading import load_data, test_training_data
from scripts.config_and_logging import generate_run_id, save_run_metadata, create_output_dir, log_to_mlflow, save_model_and_pipeline, run_with_config, load_config_hydra
from scripts.model_pipeline_cv2 import choose_best_model
from sklearn.metrics import root_mean_squared_error, make_scorer

from pathlib import Path
import logging

# Basic logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Get the project root directory
project_root = Path(__file__).parent
config_path = project_root / "configs"


def main(config_name="config_cv.yaml"):
    config = load_config_hydra(config_name=config_name, config_path=str(config_path))

    print("Load data...")
    train_df, _, _ = load_data(config)

    test_training_data(train_df)

    print("Choose best models on training and validation set. Retrain on on full training set...")

    # Model selection
    best_model_results = choose_best_model(
        None,
        train_df, 
        config
    )

    print(best_model_results)


if __name__ == "__main__":
    main()


if False: 
    from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, TimeSeriesSplit
    from scripts.preprocessing import get_pipeline_for_model
    from scripts.models import get_model
    from sklearn.metrics import make_scorer

    # Get the project root director
    config = load_config_hydra(config_name="config.yaml", config_path=str(config_path))
        

    train_df, _, _ = load_data(config)
    model_config = config['models'][1]
    pipeline = get_pipeline_for_model(model_config, config)
    model = get_model(model_config["type"], model_config['params'])

    y_train = train_df[config['data']['target_column']]
    X_train = train_df.drop(columns=[config['data']['target_column']])
    X_train_processed = pipeline.fit_transform(X_train)

    cv = TimeSeriesSplit(n_splits=5)
    rmse_score = make_scorer(root_mean_squared_error, response_method='predict', greater_is_better=False)
    search = GridSearchCV(
            estimator=model,
            param_grid={'alpha': [0.1, 0.5]},
            cv=cv,
            scoring=rmse_score,
            n_jobs=1,
            verbose=2
        )
    search.fit(X_train_processed, y_train)
    print("Best score from GridSearchCV:", -search.best_score_)
    print("Best parameters from GridSearchCV:", search.best_params_)