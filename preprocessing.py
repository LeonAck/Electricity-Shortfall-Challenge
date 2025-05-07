import pandas as pd

def preprocess_data(df):

    # Ensure time is datetime type
    df['time'] = pd.to_datetime(df['time'])

    # Convert Valencia_wind_deg to numerical values
    df['Valencia_wind_deg_cat'] = df['Valencia_wind_deg'].astype(str).str.replace('level_', '').astype(int)

    # Convert Seville_pressure to numerical values
    df['Seville_pressure_cat'] = df['Seville_pressure'].astype(str).str.replace('sp', '').astype(int)

    return df


