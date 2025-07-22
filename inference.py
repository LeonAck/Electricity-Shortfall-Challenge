import joblib
import pandas as pd

from preprocessing import * 

def load_models():
    """Load saved models if they exist, otherwise return None"""
    try:
        model = joblib.load("saved_models/best_model.pkl")
        pipeline = joblib.load("saved_models/preprocessing_pipeline.pkl")
        return model, pipeline
    except FileNotFoundError:
        print("Warning: Saved models not found. Train models first.")
        return None, None

def predict_batch(input_df: pd.DataFrame, model=None, pipeline=None) -> pd.Series:
    if model is None or pipeline is None:
        model, pipeline = load_models()
        if model is None or pipeline is None:
            raise ValueError("Models not available. Please train models first.")
    
    X_transformed = pipeline.transform(input_df)
    preds = model.predict(X_transformed)
    return pd.Series(preds, index=input_df.index)

def predict_single(input_row: pd.Series, model=None, pipeline=None) -> float:
    if model is None or pipeline is None:
        model, pipeline = load_models()
        if model is None or pipeline is None:
            raise ValueError("Models not available. Please train models first.")
    
    df_row = pd.DataFrame([input_row])
    X_transformed = pipeline.transform(df_row)
    pred = model.predict(X_transformed)[0]
    return pd.Series(pred)


if __name__ == "__main__":
    # Example test data
    df = pd.read_csv("Data/Data_raw/df_test.csv")
    preds = predict_batch(df)
    preds.to_csv("Output/predictions.csv")
    print("✅ Predictions saved.")