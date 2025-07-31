import hydra
from omegaconf import OmegaConf, DictConfig
import random
from pathlib import Path

# Get the project root directory
project_root = Path(__file__).parent
config_path = project_root / "configs/config.yaml"

def run_with_config(cfg: DictConfig) -> dict:
    """The actual logic without Hydra decoration"""
    return dict(cfg)

@hydra.main(version_base=None, config_path=str(config_path), config_name="config")
def hydra_main(cfg: DictConfig) -> None:
    return run_with_config(cfg)

def another_function():
    # Now you can call the logic directly if you have a config
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(config_path)
    result = run_with_config(cfg)
    return result

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    # Set seed
    random.seed(cfg.seed)
    print(dict(cfg))

    # Access models
    for i, model_cfg in enumerate(cfg["models"]):
        print(model_cfg)
        print(f"Model {i+1}: {model_cfg['type']}, scaling={model_cfg['scaling']}")


    return dict(cfg)

    # Your training logic here...

if __name__ == "__main__":
    cfg = main()
    print("ANOTHER FUNCTION", another_function())