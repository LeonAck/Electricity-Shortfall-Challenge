
import pandas as pd
import os

def load_data(config):
    data_dir = os.path.join(os.getcwd(), config['data']['data_path'])
    train_df = pd.read_csv(data_dir + config['data']['train_path'])
    test_df = pd.read_csv(data_dir + config['data']['test_path'])
    sample_submission = pd.read_csv(data_dir + config['data']['submission_path'])
    return train_df, test_df, sample_submission
