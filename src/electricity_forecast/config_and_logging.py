import yaml
import os
from datetime import datetime
import json
import joblib
import mlflow
import mlflow.sklearn  # For sklearn models
from sklearn.pipeline import Pipeline
import hydra
from omegaconf import DictConfig
from pathlib import Path

    
def run_with_config(cfg: DictConfig) -> dict:
    """The actual logic without Hydra decoration"""
    return dict(cfg)

def load_config_hydra(config_name="config", config_path="C:/Users/lackerman008/Electricity Shortfall Challenge/configs"):
    config_dir = Path(config_path).absolute()
    
    with hydra.initialize_config_dir(config_dir=str(config_dir), version_base=None):
        # This gives you the FULL Hydra config, same as in hydra_main
        cfg = hydra.compose(config_name)
        result = run_with_config(cfg)
        return result

# Get the project root directory
project_root = Path(__file__).parent.parent
config_path = project_root / "configs"

@hydra.main(version_base=None, config_path=str(config_path), config_name="config")
def hydra_main(cfg: DictConfig) -> DictConfig:
    print(cfg)
    return run_with_config(cfg)

def generate_run_id(config):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = config["run"].get("run_name", "run")
    return f"{run_name}_{timestamp}"

def create_output_dir(run_name, run_id, config):
    output_dir = os.path.join(f"{config['output']['saved_models_folder']}/{run_name}/", run_id)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def save_run_metadata(output_dir, config, metrics):
    with open(os.path.join(output_dir, "config_used.yaml"), "w") as f:
        yaml.dump(config, f)
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

def setup_mlflow(config):
    mlflow_cfg = config.get("mlflow", {})
    if not mlflow_cfg.get("enabled", False):
        return None
    mlflow.set_tracking_uri(mlflow_cfg.get("tracking_uri", "file:./mlruns"))
    mlflow.set_experiment(mlflow_cfg.get("experiment_name", "default"))
    return mlflow_cfg

def extract_model_params(config_models_list, model_name):
    """Zoek de juiste params uit de YAML-config-lijst obv. naam."""
    for model in config_models_list:
        if model["type"] == model_name:
            return model.get("params", {})
    return {}


def log_to_mlflow(config, output_dir, run_id, model_name, model_object, metrics, parameters=None):
    mlflow_cfg = setup_mlflow(config)
    if not mlflow_cfg:
        return

    with mlflow.start_run(run_name=run_id):
        
        mlflow.log_param("model", model_name)
        # Alleen numerieke metrics loggen
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(key, value)
            else:
                mlflow.set_tag(f"meta_{key}", str(value))  # log strings als tag

        # Log parameters
        model_params = extract_model_params(config.get("models", []), model_name)
        if isinstance(model_params, dict):
            mlflow.log_params(model_params)

        # Log model (skip if AutoReg or not sklearn)
        if model_name != "AR1":
            try:
                mlflow.sklearn.log_model(model_object, name="model")
            except Exception as e:
                print(f"MLflow model logging failed: {e}")

        # Log config used
        mlflow.log_artifact(f"{output_dir}/config_used.yaml")
        # Log plot if it exists
        plot_path = f"{output_dir}/{model_name}_validation_plot.png"
        if os.path.exists(plot_path):
            mlflow.log_artifact(plot_path)

        else:
            print("Plot does not exist, skipping logging of validation plot.")
            print("plot_path", plot_path)

def save_model_and_pipeline(pipeline, model, config):
    os.makedirs(config['output']['saved_models_folder'], exist_ok=True)
    if config["run"]["gcloud"]:

        full_pipeline = Pipeline([
            ("preprocessing", pipeline),  
            ("model", model)              
        ])

        # Save combined artifact
        joblib.dump(full_pipeline, f"{config['output']['saved_models_folder']}/{config['output']['combined_model_filename']}")
        
    else: 
        joblib.dump(pipeline, f"{config['output']['saved_models_folder']}/{config['output']['saved_pipeline_filename']}")
        joblib.dump(model, f"{config['output']['saved_models_folder']}/{config['output']['saved_models_filename']}")

def store_train_features(train_df, config):
    os.makedirs(config['output']['saved_models_folder'], exist_ok=True)
    train_df.to_csv(f"{config['output']['saved_models_folder']}/train_features.csv", index=False)
