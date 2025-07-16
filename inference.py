import joblib
import pandas as pd

from preprocessing import * 

# Load bundle
model = joblib.load("saved_models/best_model.pkl")
pipeline = joblib.load("saved_models/preprocessing_pipeline.pkl")

def predict_batch(input_df: pd.DataFrame) -> pd.Series:
    X_transformed = pipeline.transform(input_df)
    preds = model.predict(X_transformed)
    return pd.Series(preds, index=input_df.index)

def predict_single(input_row: pd.Series) -> float:
    df_row = pd.DataFrame([input_row])
    X_transformed = pipeline.transform(df_row)
    pred = model.predict(X_transformed)[0]
    return pd.Series(pred)


if __name__ == "__main__":
    # Example test data
    df = pd.read_csv("Data/Data_raw/df_test.csv")
    preds = predict_batch(df)
    preds.to_csv("predictions.csv")
    print("✅ Predictions saved.")