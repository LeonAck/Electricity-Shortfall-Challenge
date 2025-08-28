# File: prefect_energy_forecast_flow.py

# At the top of your flow file
from pathlib import Path

# Add the project root to Python path
#sys.path.append(str(Path(__file__).parent.parent.parent))


from prefect import flow, task, get_run_logger
from prefect.tasks import task
from prefect.runtime import flow_run

from datetime import datetime
from pathlib import Path
import joblib
import json
import os
import tempfile
from google.cloud import storage
from typing import Dict, Any,  Optional

# Import your modules (ensure they're in Python path)
from src.electricity_forecast.data_loading import load_data, test_training_data
from src.electricity_forecast.config_and_logging import load_config_hydra
from src.electricity_forecast.train import choose_best_model

# =============================================================================
# CONFIGURATION
# =============================================================================

GCP_PROJECT_ID = "energy-forecast-467113"
MODEL_BUCKET = "forecast_bucket_inference"
SERVICE_ACCOUNT_EMAIL = "cloud-run-sa-energy-forecast-i@energy-forecast-467113.iam.gserviceaccount.com"
CONFIG_NAME = "config_cv.yaml"

# Prefect auto-injects logger
@task
def get_project_paths() -> Dict[str, str]:
    """Get project directory paths"""
    logger = get_run_logger()
    project_root = Path(__file__).parent.parent.parent
    config_path = project_root / "configs"

    paths = {
        'project_root': str(project_root),
        'config_path': str(config_path),
        'config_name': CONFIG_NAME
    }
    logger.info("Paths set: %s", paths)
    return paths


@task
def validate_energy_data_quality(paths: Dict[str, str]) -> Dict[str, Any]:
    """Validate energy data quality"""
    logger = get_run_logger()
    logger.info("Loading and validating energy forecast training data...")

    config = load_config_hydra(
        config_name=paths['config_name'],
        config_path=paths['config_path']
    )

    train_df, _, _ = load_data(config)
    test_training_data(train_df)  # Raises if fails

    metrics = {
        'train_size': len(train_df),
        'validation_date': datetime.now().isoformat()
    }
    logger.info("Data validation passed. Training size: %d", metrics['train_size'])
    return metrics


@task
def train_energy_forecast_model(
    paths: Dict[str, str],
    data_metrics: Dict[str, Any]
) -> Dict[str, Any]:
    """Train and select best model"""
    logger = get_run_logger()
    config = load_config_hydra(
        config_name=paths['config_name'],
        config_path=paths['config_path']
    )

    config['run']['trigger'] = 'prefect_scheduled'
    execution_date = datetime.now().strftime("%Y-%m-%d")

    logger.info("Loading training data...")
    train_df, _, _ = load_data(config)

    logger.info("Starting model training and selection...")
    best_model_results = choose_best_model(train_df, config)

    temp_model_path = f"temp_model_{execution_date.replace('-', '')}.joblib"
    joblib.dump(best_model_results.full_pipeline, temp_model_path)

    model_results = {
        'model_name': best_model_results.model_name,
        'cv_rmse': best_model_results.cv_rmse,
        'is_from_mlflow': best_model_results.is_from_mlflow,
        'production_version': best_model_results.production_version,
        'temp_model_path': temp_model_path,
        'model_performance': {
            'cv_rmse': best_model_results.cv_rmse,
            'data_size': data_metrics['train_size'],
            'training_date': execution_date,
            'model_type': best_model_results.model_name
        }
    }

    logger.info(f"Model selected: {model_results['model_name']} | RMSE: {model_results['cv_rmse']:.4f}")
    return model_results


@task
def check_model_improvement(model_results: Dict[str, Any]) -> bool:
    """Decide whether to upload based on improvement"""
    logger = get_run_logger()

    if model_results['is_from_mlflow']:
        logger.info("Using existing MLflow model. Skipping upload.")
        return False

    logger.info(f"New model RMSE: {model_results['cv_rmse']:.4f} → Will upload to GCS.")
    return True


@task
def upload_model_to_gcs(model_results: Dict[str, Any]) -> str:
    """Upload model and metadata to GCS"""
    logger = get_run_logger()
    client = storage.Client(project=GCP_PROJECT_ID)
    bucket = client.bucket(MODEL_BUCKET)

    version_tag = datetime.now().strftime("%Y%m%d")
    gcs_model_dir = f"models/energy-forecast-model/{version_tag}/"

    temp_model_path = model_results['temp_model_path']
    model_pipeline = joblib.load(temp_model_path)

    with tempfile.TemporaryDirectory() as tmp_dir:
        # Save artifacts locally
        model_path = os.path.join(tmp_dir, "model.joblib")
        joblib.dump(model_pipeline, model_path)

        metadata = {
            'model_name': model_results['model_name'],
            'cv_rmse': model_results['cv_rmse'],
            'training_date': model_results['model_performance']['training_date'],
            'version': version_tag,
            'data_size': model_results['model_performance']['data_size'],
            'model_type': model_results['model_performance']['model_type'],
            'mlflow_version': model_results['production_version'],
            'prefect_flow_run_id': flow_run.get_id(),
            'execution_date': datetime.now().isoformat()
        }
        metadata_path = os.path.join(tmp_dir, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        model_info = {
            'latest_model': f"gs://{MODEL_BUCKET}/{gcs_model_dir}model.joblib",
            'performance': {
                'cv_rmse': model_results['cv_rmse'],
                'training_samples': model_results['model_performance']['data_size']
            },
            'created_at': datetime.now().isoformat(),
            'version': version_tag
        }
        model_info_path = os.path.join(tmp_dir, "model_info.json")
        with open(model_info_path, 'w') as f:
            json.dump(model_info, f, indent=2)

        # Upload files
        for local, remote in [
            (model_path, f"{gcs_model_dir}model.joblib"),
            (metadata_path, f"{gcs_model_dir}metadata.json"),
            (model_info_path, f"{gcs_model_dir}model_info.json"),
            (model_info_path, "models/energy-forecast-model/latest/model_info.json")
        ]:
            blob = bucket.blob(remote)
            blob.upload_from_filename(local)
            logger.info(f"Uploaded {remote}")

    # Cleanup
    if os.path.exists(temp_model_path):
        os.remove(temp_model_path)

    gcs_path = f"gs://{MODEL_BUCKET}/{gcs_model_dir}"
    logger.info(f"✅ Model uploaded to {gcs_path}")
    return gcs_path

@task
def run_upload_task(model_results: Dict[str, Any]) -> Optional[str]:
    try:
        result = upload_model_to_gcs(model_results)
        return result
    except Exception as exc:
        get_run_logger().warning(f"Upload task failed: {exc}")
        return None

@task
def run_validation_task(gcs_path: str) -> bool:
    try:
        validate_gcs_upload(gcs_path)
        return True
    except Exception as exc:
        get_run_logger().warning(f"Validation failed: {exc}")
        return False


@task
def validate_gcs_upload(model_gcs_path: str) -> bool:
    """Validate all required files exist in GCS"""
    logger = get_run_logger()
    client = storage.Client(project=GCP_PROJECT_ID)
    bucket = client.bucket(MODEL_BUCKET)

    version_tag = datetime.now().strftime("%Y%m%d")
    gcs_model_dir = f"models/energy-forecast-model/{version_tag}/"

    required_files = [
        f"{gcs_model_dir}model.joblib",
        f"{gcs_model_dir}metadata.json",
        f"{gcs_model_dir}model_info.json"
    ]

    all_exist = True
    for file_path in required_files:
        blob = bucket.blob(file_path)
        if blob.exists():
            size = blob.size
            logger.info(f"✅ {file_path} exists ({size} bytes)")
        else:
            logger.error(f"❌ {file_path} does not exist")
            all_exist = False

    if all_exist:
        logger.info("✅ All files validated in GCS.")
    else:
        raise ValueError("❌ One or more files missing in GCS.")

    return all_exist


@task(name="Send Notification")
def send_notification(model_results: Dict[str, Any], upload_success: bool = True, model_gcs_path: str = None):
    """Send final notification via log (extend to Slack/email later)"""
    logger = get_run_logger()
    version_tag = datetime.now().strftime("%Y%m%d")

    if model_results['is_from_mlflow']:
        message = f"""
🔋 Energy Forecast ML Pipeline - No Update Needed
📅 Execution Date: {model_results['model_performance']['training_date']}
🤖 Current Best Model: {model_results['model_name']}
📊 Performance (CV RMSE): {model_results['cv_rmse']:.4f}
ℹ️  Status: Existing MLflow model is still the best
📈 Data Size: {model_results['model_performance']['data_size']} samples
No new model uploaded to GCS.
        """
    else:
        message = f"""
🔋 Energy Forecast ML Pipeline - Model Updated!
📅 Execution Date: {model_results['model_performance']['training_date']}
🤖 New Model: {model_results['model_name']}
📊 Performance (CV RMSE): {model_results['cv_rmse']:.4f}
🚀 Upload Status: {'✅ Success' if upload_success else '❌ Failed'}
🏷️ Version: {version_tag}
📈 Data Size: {model_results['model_performance']['data_size']} samples
📍 Model Location: {model_gcs_path if model_gcs_path else 'Upload failed'}
📊 MLflow Run: {model_results['production_version']}
        """
    logger.info(message)


# =============================================================================
# PREFECT FLOW DEFINITION
# =============================================================================

@flow(
    name="Energy Forecast ML Pipeline",
    description="Train, evaluate, and upload model to GCS if improved",
    timeout_seconds=2 * 3600  # 2 hours
)
def energy_forecast_flow():
    
    # Setup
    logger = get_run_logger()
    paths = get_project_paths()
    data_metrics = validate_energy_data_quality(paths)
    

    # Train model
    model_results = train_energy_forecast_model(paths, data_metrics)
    should_upload = check_model_improvement(model_results)

    # Conditional upload
    upload_result = None
    validation_result = None

    upload_result = None
    validation_result = None

        
    if should_upload:
        logger.info("Uploading new model to GCS...")
        try:
            upload_result = run_upload_task(model_results)
            if upload_result is not None:
                logger.info("Upload succeeded. Validating...")
                validation_result = run_validation_task(upload_result)
            else:
                logger.warning("Upload returned None — treating as failure.")
                validation_result = False
        except Exception as e:
            logger.warning(f"Upload task encountered an error: {e}")
            upload_result = None
            validation_result = False
    else:
        logger.info("No upload needed — using existing production model.")
        upload_result = None
        validation_result = False

        # Always send notification
        send_notification(
            model_results=model_results,
            upload_success=(validation_result is True),
            model_gcs_path=upload_result
        )


# =============================================================================
# HOW TO RUN & SCHEDULE
# =============================================================================

if __name__ == "__main__":
    # ✅ Run locally once
    energy_forecast_flow()
    
    # client = storage.Client(project=GCP_PROJECT_ID)
    # bucket = client.bucket(MODEL_BUCKET)
    # buckets = client.list_buckets()
    #for b in buckets:
        # print(b.name)
    # ✅ Schedule to run weekly
    from prefect.tasks import task
    from prefect.server.schemas.schedules import IntervalSchedule
    from datetime import timedelta

    # Optional: Uncomment to deploy as scheduled flow
    # energy_forecast_flow.schedule = IntervalSchedule(
    #     interval=timedelta(weeks=1),
    #     anchor_date=datetime(2024, 1, 1)
    # )