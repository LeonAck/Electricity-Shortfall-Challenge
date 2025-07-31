
import pandas as pd
import os
from saved_models.data_validation_model import PredictionRequest

def load_data(config):
    """
    Load training and test data based on the configuration provided.
    Args:
        config (dict): Configuration dictionary containing paths to data files.
    Returns:
        tuple: DataFrames for training, test, and sample submission.
    """
    data_dir = os.path.join(os.getcwd(), config['data']['data_path'])
    train_df = pd.read_csv(data_dir + config['data']['train_path'])
    test_df = pd.read_csv(data_dir + config['data']['test_path'])
    sample_submission = pd.read_csv(data_dir + config['data']['submission_path'])
    
    return train_df, test_df, sample_submission

def test_training_data(train_df):
    
    try:
        for _, row in train_df.iterrows():
            features_dict = {"features": row.to_dict()}
            request_data = PredictionRequest(**features_dict)

        print("✅ Data test successful")
    except Exception as e:
        print(f"❌ Error during data testing: {e}")
        
