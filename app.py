# app.py (save in your repo root)
from flask import Flask, request, jsonify
import joblib
from google.cloud import storage
import os
import pandas as pd
from saved_models.data_validation_model import PredictionRequest  # ← Import here

app = Flask(__name__)

# Load model at startup (Cloud Run initializes container on first request)
MODEL_VERSION = os.getenv("MODEL_VERSION", "v1")  # Get version from env
BUCKET_NAME = "forecast_bucket_inference"

def load_model():
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(f"models/{MODEL_VERSION}/combined_model.joblib")
    blob.download_to_filename("/tmp/model.joblib")
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
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 8080)))