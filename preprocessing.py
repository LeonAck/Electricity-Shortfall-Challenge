from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np

def preprocess_data(df):

    # Convert Valencia_wind_deg to numerical values
    df['Valencia_wind_deg_cat'] = df['Valencia_wind_deg'].astype(str).str.replace('level_', '').astype(int)
    df = df.drop(columns=['Valencia_wind_deg'])
    # Convert Seville_pressure to numerical values
    df['Seville_pressure_cat'] = df['Seville_pressure'].astype(str).str.replace('sp', '').astype(int)
    df = df.drop(columns=['Seville_pressure'])
    # Ensure time is datetime type
    df = set_datetime_as_index(df)

    # Remove any unwanted columns
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])

    return df

def set_datetime_as_index(df, freq='3h', fill_method='interpolate'):
    """
    Convert 'time' column to datetime index, reindex to regular intervals,
    and impute missing rows.

    Parameters:
    - df: pd.DataFrame with a 'time' column
    - freq: str, frequency string for datetime (default='3H')
    - fill_method: str, one of ['interpolate', 'ffill', 'bfill', 'zero']

    Returns:
    - DataFrame with datetime index at regular intervals and missing values filled
    """
    import pandas as pd

    # Convert and sort time
    df['time'] = pd.to_datetime(df['time'])
    df = df.set_index('time')

    # Create complete datetime index
    full_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq=freq)

    # Reindex
    df = df.reindex(full_index)

    # Impute missing values
    if fill_method == 'interpolate':
        df = df.interpolate()
    elif fill_method == 'ffill':
        df = df.ffill()
    elif fill_method == 'bfill':
        df = df.bfill()
    elif fill_method == 'zero':
        df = df.fillna(0)
    else:
        raise ValueError("Unsupported fill_method. Choose from ['interpolate', 'ffill', 'bfill', 'zero'].")

    df = df.asfreq('3h')

    return df


# works
class ValenciaPressureImputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        df = X.copy()
        df['hour'] = df.index.hour
        df['day'] = df.index.dayofweek
        df['month'] = df.index.month

        self.group_medians_ = df.groupby(['hour', 'day', 'month'])['Valencia_pressure'].median()
        return self

    def transform(self, X):
        df = X.copy()
        df['hour'] = df.index.hour
        df['day'] = df.index.dayofweek
        df['month'] = df.index.month

        def impute(row):
            if pd.isna(row['Valencia_pressure']):
                return self.group_medians_.get((row['hour'], row['day'], row['month']), np.nan)
            return row['Valencia_pressure']

        df['Valencia_pressure'] = df.apply(impute, axis=1)
        df.drop(['hour', 'day', 'month'], axis=1, inplace=True)
        return df

def check_nan_in_column(df, column):
    """
    Checks the number of NaN values in a specific column.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame to analyze
    column : str
        The column name to check
        
    Returns:
    --------
    tuple
        (nan_count, nan_percentage)
    """
    nan_count = df[column].isna().sum()
    nan_percentage = (nan_count / len(df)) * 100
    
    print(f"\nNaN Analysis for column '{column}':")
    print(f"Number of NaN values: {nan_count}")
    print(f"Percentage of NaN values: {nan_percentage:.2f}%")
    
    return nan_count, nan_percentage


def undiff_series(diffed_array, starting_value):

    return diffed_array.cumsum() + starting_value
