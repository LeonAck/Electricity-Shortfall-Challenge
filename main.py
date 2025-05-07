from data_loading import load_data
from data_exploration import initial_checks

train_df, test_df, sample_submission = load_data()

initial_checks([train_df, test_df, sample_submission], ['train_df', 'test_df', 'sample_submission'])

