# ⚡ Electricity Load Shortfall Forecasting

**Electricity-Shortfall-Challenge** is a machine learning system that forecasts the **3-hour electricity load shortfall in Spain**—the gap between renewable energy generation and fossil fuel-based supply—using city-level weather data from **2015–2017**. This project supports energy infrastructure planning by modeling renewable supply variability under changing weather conditions.

---

## 📌 Table of Contents
- [Project Overview](#-project-overview)
- [Dataset](#-dataset)
- [Modeling Goal](#-modeling-goal)
- [Architecture](#-architecture)
- [File Structure](#-file-structure)
- [Installation & Setup](#-installation--setup)
- [Training Pipeline](#-training-pipeline)
- [Prefect Workflow](#-prefect-workflow)
- [Model Evaluation & Selection](#-model-evaluation--selection)
- [MLflow Integration](#-mlflow-integration)
- [API Deployment with Flask](#-api-deployment-with-flask)
- [CI/CD with Google Cloud Build](#-cicd-with-google-cloud-build)
- [Next Steps](#-next-steps)

---

## 📊 Project Overview
- **Objective**: Forecast electricity load shortfall using weather data.
- **Scope**: 5 Spanish cities, 2015–2017.
- **MLOps Pipeline**: Data validation → model selection → training → cloud deployment → API inference.

---

## 📂 Dataset
- **Source**: Anonymized historical weather and energy data.
- **Features**: City-level weather variables (wind speed, humidity, etc.).
- **Target**: 3-hour electricity load shortfall.

---

## 🎯 Modeling Goal
- Predict the shortfall with high accuracy to optimize energy infrastructure planning.
- Compare models using **cross-validated RMSE** and hyperparameter tuning.

---

## 🏗️ Architecture
- **Orchestration**: Prefect for workflow automation.
- **Modeling**: Scikit-learn, XGBoost, MLflow for tracking.
- **Deployment**: Flask API on Google Cloud Run.
- **Storage**: Google Cloud Storage for models and metadata.
- **CI/CD**: Google Cloud Build for automated testing and deployment.

---

## 📁 File Structure

<custom-element data-json="%7B%22type%22%3A%22table-metadata%22%2C%22attributes%22%3A%7B%22title%22%3A%22File%20Structure%22%7D%7D" />

| Path | Description |
|------|-------------|
| `configs/` | Configuration files (logging, model selection, etc.) |
| `data/` | Raw and processed data (`df_train.csv`, `df_test.csv`, `processed_data.parquet`) |
| `notebooks/` | Exploration and analysis notebooks |
| `prefect/flows/` | Prefect orchestration scripts |
| `saved_models/` | Pydantic schema for API validation |
| `src/electricity_forecast/` | Core logic (training, preprocessing, models) |
| `tests/` | Unit and integration tests |
| `app.py` | Flask API for predictions |
| `cloudbuild.yaml` | CI/CD pipeline configuration |
| `Dockerfile` | Containerization for Cloud Run |
| `pyproject.toml` | Project dependencies |

---

### Clone and Set Up
```bash
git clone https://github.com/LeonAck/Electricity-Shortfall-Challenge.git
cd Electricity-Shortfall-Challenge
```

### Install  `uv` and dependencies
```bash
pip install uv
uv sync
```
This installs all required packages including: scikit-learn, xgboost, mlflow, hydra-core, prefect, flask, google-cloud-storage.

### Set Up MLflow (Optional Local Tracking)
```bash
mlflow ui --backend-store-uri mlruns/
```

## 🚀 Training Pipeline

The main training logic is in `src/electricity_forecast/train.py`. It:
- Loads training data
- Applies preprocessing (scaling, imputation)
- Evaluates multiple models via cross-validation
- Performs hyperparameter tuning (if enabled)
- Compares with current production model in MLflow
- Retrains and registers the best model

```python
from src.electricity_forecast.train import choose_best_model
best_result = choose_best_model(train_df, config)
```

## 📈 Model Evaluation & Selection

- **Cross-validation**: TimeSeriesSplit or custom time-aware splits
- **Hyperparameter tuning**: GridSearchCV via config
- **Model factory**: `get_model()` supports 15+ regressors
- **Final decision**: New model only replaces production model if it improves CV-RMSE

## 🧪 MLflow Integration

- Logs all model experiments (RMSE, params, metrics)
- Registers best model in MLflow Model Registry
- Promotes winning version to production alias
- Enables comparison with existing best model

## 🌐 API Deployment with Flask

The model is served via a lightweight Flask API deployed on Google Cloud Run.

### Endpoints
- `POST /predict`: Make predictions
- `GET /health`: Health check (used by Cloud Run)

### Request Format
```json
{
  "features": {
    "time": "2017-01-01 00:00:00",
    "Madrid_wind_speed": 12.5,
    "Barcelona_humidity": 80,
    ...
  }
}
```

### Response
```json
{ "prediction": [234.5] }
```

### Deploy to Cloud Run
```bash
gcloud run deploy energy-forecast-api \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```
Automatically loads the latest model from GCS via `latest/model_info.json`.

## 🛠️ CI/CD with Google Cloud Build

Automated testing via `test_code.yml` and cloud build via `gcp-build-key.yml`.

### Workflow:
- Run pytest on push
- Validate data and model training
- (Optional) Deploy API or trigger Prefect flow

## 🔮 Next Steps

- Finish testing for pipeline and training
- Add model monitoring (Evidently, WhyLogs) for drift & performance
- Set up alerting on model degradation

💡 Want to contribute? Open an issue or submit a PR!
