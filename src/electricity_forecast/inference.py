import joblib
import pandas as pd
import traceback

from src.preprocessing import * 

def load_models(folder='saved_models'):
    """Load saved models if they exist, otherwise return None"""
    try:
        model = joblib.load(f"{folder}/best_model.pkl")
        pipeline = joblib.load(f"{folder}/preprocessing_pipeline.pkl")
        return model, pipeline
    except FileNotFoundError:
        print(f"Warning: Model or pipeline not found in folder '{folder}'. Train models first.")
    except Exception as e:
        print(f"Error loading models: {e}")
        traceback.print_exc()
    return None, None


def predict_batch(input_df: pd.DataFrame, model=None, pipeline=None) -> pd.Series:
    """
    Predict using the trained model and preprocessing pipeline.
    Args:
        input_df (pd.DataFrame): DataFrame containing the input features.
        model: Trained model object (optional, will load if None).
        pipeline: Preprocessing pipeline object (optional, will load if None).
    Returns:
        pd.Series: Predictions for the input DataFrame.
    """
    try:
        if model is None or pipeline is None:
            model, pipeline = load_models()
            if model is None or pipeline is None:
                raise ValueError("Models not available. Please train models first.")
        
        X_transformed = pipeline.transform(input_df)
        preds = model.predict(X_transformed)
        return pd.Series(preds, index=input_df.index)
    except Exception as e:
        print(f"Error during batch prediction: {e}")
        traceback.print_exc()
        raise

def predict_single(input_row: pd.Series, model=None, pipeline=None) -> float:
    """
    Predict a single row of input data.
    Args:
        input_row (pd.Series): A single row of input features.
    Returns:
        pd.Series: Prediction for the input row.
    """
    try:
        if model is None or pipeline is None:
            model, pipeline = load_models()
            if model is None or pipeline is None:
                raise ValueError("Models not available. Please train models first.")
        
        df_row = pd.DataFrame([input_row])
        X_transformed = pipeline.transform(df_row)
        pred = model.predict(X_transformed)[0]
        return pd.Series(pred)
    except Exception as e:
        print(f"Error during single-row prediction: {e}")
        traceback.print_exc()
        raise

if __name__ == "__main__":
    # Example test data
    df = pd.read_csv("data/data_raw/df_test.csv")
    preds = predict_batch(df)
    preds.to_csv("output/predictions.csv")
    print("Predictions saved.")