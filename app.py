# app.py (save in your repo root)
from flask import Flask, request, jsonify
import joblib
from google.cloud import storage
import os

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
    data = request.json
    # Add input validation here (MLOps best practice!)
    prediction = model.predict([data["features"]])
    return jsonify({"prediction": prediction.tolist()})

@app.route("/health", methods=["GET"])
def health():
    return "OK", 200  # Cloud Run uses this for readiness probes

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 8080)))