# app.py (save in your repo root)
from flask import Flask, request, jsonify
import joblib
from google.cloud import storage
import os
import pandas as pd
from src.electricity_forecast.data_validation_model import PredictionRequest  # ← Import here
import json

app = Flask(__name__)

# Load model at startup (Cloud Run initializes container on first request)
MODEL_VERSION = os.getenv("MODEL_VERSION", "v1")  # Get version from env
BUCKET_NAME = "forecast_bucket_inference"

def load_model():
    """Load the latest production model from GCS (via latest/model_info.json)"""
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)

    # Step 1: Download model_info.json to find latest model
    info_blob = bucket.blob("models/energy-forecast-model/latest/model_info.json")
    info_blob.download_to_filename("/tmp/model_info.json")

    with open("/tmp/model_info.json", "r") as f:
        model_info = json.load(f)

    # Step 2: Extract model path
    latest_model_gcs_path = model_info["latest_model"]  # "gs://bucket/..."
    # Convert to blob path
    blob_path = latest_model_gcs_path.replace(f"gs://{BUCKET_NAME}/", "")

    # Step 3: Download and load model
    model_blob = bucket.blob(blob_path)
    model_blob.download_to_filename("/tmp/model.joblib")

    return joblib.load("/tmp/model.joblib")

model = load_model()

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Parse and validate input using Pydantic
        request_data = PredictionRequest(**request.json)

        # Convert to DataFrame
        features_dict = request_data.features.model_dump()
        df = pd.DataFrame([features_dict])

        

        # Make prediction
        prediction = model.predict(df)

        return jsonify({"prediction": prediction.tolist()})

    except Exception as e:
        # Handle both Pydantic validation errors and others
        if isinstance(e, ValueError) or "ValidationError" in str(type(e)):
            return jsonify({"error": "Invalid input", "details": str(e)}), 400
        else:
            app.logger.error(f"Prediction error: {e}")
            return jsonify({"error": "Internal server error"}), 500

@app.route("/health", methods=["GET"])
def health():
    return "OK", 200  # Cloud Run uses this for readiness probes

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(
        host="0.0.0.0",  # Listen on all interfaces
        port=port,       # Use PORT environment variable
        debug=False      # Never enable debug in production
    )
