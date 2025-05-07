import pandas as pd

def preprocess_data(df):

    # Ensure time is datetime type
    train_df['time'] = pd.to_datetime(train_df['time'])

    # Convert Valencia_wind_deg to numerical values
    df['Valencia_wind_deg'] = df['Valencia_wind_deg'].astype(str).str.replace('level_', '').astype(int)

    # Convert Seville_pressure to numerical values
    df['Seville_pressure'] = df['Seville_pressure'].astype(str).str.replace('sp', '').astype(int)

    return df


