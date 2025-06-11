import yaml
import os
from datetime import datetime
import json

def load_config(config_path="Configs/baseline.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def generate_run_id(config):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = config["run"].get("run_name", "run")
    return f"_{timestamp}"

def create_output_dir(run_name, run_id):
    output_dir = os.path.join(f"Output/{run_name}/", run_id)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def save_run_metadata(output_dir, config, metrics):
    with open(os.path.join(output_dir, "config_used.yaml"), "w") as f:
        yaml.dump(config, f)
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)