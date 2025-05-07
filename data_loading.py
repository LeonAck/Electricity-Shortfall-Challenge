import pandas as pd
import os

def load_data():
    data_dir = os.path.join(os.getcwd(), 'Data/Data_raw')
    train_df = pd.read_csv(data_dir + '/df_train.csv')
    test_df = pd.read_csv(data_dir + '/df_test.csv')
    sample_submission = pd.read_csv(data_dir + '/sample_submission_load_shortfall (1).csv')
    return train_df, test_df, sample_submission
