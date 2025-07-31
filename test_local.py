import joblib
import json
import pandas as pd
from saved_models.data_validation_model import PredictionRequest  # Import your Pydantic model

# Load model
model = joblib.load("saved_models/combined_model.joblib")  # or from GCS
check_data = True # Set to False if you want to skip validation
# Load test input
with open("test_input.json") as f:
    data = json.load(f)

if check_data:
    request_data = PredictionRequest(**data)
    # Convert to DataFrame
    features_dict = request_data.features.model_dump(by_alias=True)
    df = pd.DataFrame([features_dict])
else:
    df = pd.DataFrame([data["features"]])

try:
    pred = model.predict(df)
    print("✅ Success! Prediction:", pred[0])
except Exception as e:
    print("❌ Failed:", str(e))