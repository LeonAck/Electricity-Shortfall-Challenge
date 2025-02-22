import pandas as pd
import os

# Check the current working directory
print("Current Working Directory: ", os.getcwd())

# Create the directory string with the location of the data directory
data_dir = os.path.join(os.getcwd(), 'Data')


# Load the datasets
train_df = pd.read_csv(data_dir + '/df_train.csv')
test_df = pd.read_csv(data_dir + '/df_test.csv')
sample_submission = pd.read_csv(data_dir + '/sample_submission_load_shortfall (1).csv')

print(train_df.head())