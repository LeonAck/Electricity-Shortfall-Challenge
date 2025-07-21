from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np


def get_pipeline_for_model(model, config):

    if model["type"] in ["AR", "MA1"]:
        # Extract parameters from the config structure
        params = model.get('params', {})
        
        # Return data transformation pipeline only
        return create_timeseries_preprocessing_pipeline(
            model_type=model["type"],
            **params  # Unpack all parameters from the config
        )
    
    elif model["type"] in [
        'LinearRegression',
        'RandomForest',
        'Ridge',
        'Lasso',
        'ElasticNet',
        'BayesianRidge',
        'SGDRegressor',
        'ExtraTreesRegressor',
        'GradientBoostingRegressor',
        'XGBRegressor',
        'DecisionTreeRegressor',
        'AdaBoostRegressor',
        'KNeighborsRegressor',
        'SVR'
        ]:
        sklearn_pipeline = create_preprocessing_pipeline(imputer=get_imputer(config), 
                                             freq=config['preprocessing']['freq'],
                                             fill_method=config['preprocessing']['fill_method'], 
                                             add_time_dummies=config['preprocessing']['add_time_dummies'], 
                                             scaling=model['scaling'])
        
        return StandardTransformerWrapper(sklearn_pipeline)

    else:
        raise ValueError(f"No preprocessing pipeline defined for model_type: {model['type']}")


def create_timeseries_preprocessing_pipeline(model_type, **kwargs):
    """Create preprocessing pipeline for time series models"""
    
    if model_type in ["MA", "MA1"]:  # Handle both MA and MA1
        return Pipeline([
            ('ma_transform', MADataTransformer(
                window=kwargs.get('window', 1),
                difference_order=kwargs.get('difference_order', 1)
            ))
        ])
    
    elif model_type == "AR":
        return Pipeline([
            ('ar_transform', ARDataTransformer(
                lags=kwargs.get('lags', 1),
                difference_order=kwargs.get('difference_order', 1),
                add_lags=kwargs.get('add_lags', True)
            ))
        ])

    
    else:
        raise ValueError(f"Unknown time series model type: {model_type}")


# Creating complete preprocessing pipelines with different imputation methods
def create_preprocessing_pipeline(imputer, freq='3h', 
                                fill_method='interpolate',
                                add_time_dummies=None, 
                                scaling=False):
    """
    Create a complete preprocessing pipeline for weather data
    
    Parameters:
    - imputer: str, one of ['month/day/hour median', 'interpolation', 'knn', 'pattern']
    - freq: str, frequency string for datetime reindexing
    - fill_method: str, method for filling fully missing rows
    - pressure_column: str, name of the pressure column to impute
    - max_gap: int, for interpolation imputer
    - n_neighbors: int, for KNN imputer
    
    Returns:
    - sklearn Pipeline object
    """
    
    # Create and return the pipeline
    steps = [ ('preprocessor', WeatherDataPreprocessor(freq=freq, fill_method=fill_method, add_time_dummies=add_time_dummies))]

    steps.append(('imputer', imputer))

    if scaling:
        steps.append(('scaler', StandardScaler()))

    # Always convert to numpy
    steps.append(('to_numpy', ToNumpyArray()))  

    return Pipeline(steps)

# Modify your existing scikit-learn pipeline creation
class StandardTransformerWrapper(BaseEstimator, TransformerMixin):
    def __init__(self, sklearn_pipeline):
        self.sklearn_pipeline = sklearn_pipeline
    
    def fit(self, X, y=None):
        self.sklearn_pipeline.fit(X, y)
        return self
    
    def transform(self, X, y):
        X_transformed = self.sklearn_pipeline.transform(X)
        return X_transformed, y
    
    def fit_transform(self, X, y):
        X_transformed = self.sklearn_pipeline.fit_transform(X, y)
        return X_transformed, y


class MADataTransformer(BaseEstimator, TransformerMixin):
    """Transformer that prepares data for MA models by making y stationary"""
    
    def __init__(self, window=1, difference_order=1):
        self.window = window  # MA window parameter
        self.difference_order = difference_order
        self.original_length = None
        
    def fit(self, X, y=None):
        """Fit method - just stores metadata, no actual fitting"""
        if X is not None:
            self.original_length = len(X)
        return self
    
    def transform(self, X, y):
        """Transform X and y for MA model requirements"""
        # Make y stationary by differencing
        y_diff = self._difference_series(y, self.difference_order)
        
        # Adjust X to match the length of differenced y
        X_adjusted = X[self.difference_order:] if X is not None else None
        
        return X_adjusted, y_diff
    
    def fit_transform(self, X, y):
        """Combined fit and transform"""
        return self.fit(X, y).transform(X, y)
    
    def _difference_series(self, series, order=1):
        """Apply differencing to make series stationary"""
        result = series.copy()
        for _ in range(order):
            result = np.diff(result)
        return result
    
    def inverse_transform_predictions(self, predictions, last_values):
        """Convert differenced predictions back to original scale"""
        result = predictions.copy()
        
        # Reverse differencing by cumulative sum
        for i in range(self.difference_order):
            if i < len(last_values):
                result = np.cumsum(np.concatenate([[last_values[-(i+1)]], result]))
                result = result[1:]  # Remove the starting value
        
        return result


class ARDataTransformer(BaseEstimator, TransformerMixin):
    """Transformer for AR models - may need different preprocessing"""
    
    def __init__(self, lags=1, difference_order=1, add_lags=True):
        self.lags = lags  # Number of AR lags
        self.difference_order = difference_order
        self.add_lags = add_lags
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y):
        # Make y stationary
        y_diff = self._difference_series(y, self.difference_order)
        
        if self.add_lags and X is not None:
            # Add lagged features for AR models
            X_with_lags = self._add_lag_features(X, y, self.lags)
            # Adjust for differencing
            X_adjusted = X_with_lags[self.difference_order:]
        else:
            X_adjusted = X[self.difference_order:] if X is not None else None
            
        return X_adjusted, y_diff
    
    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X, y)
    
    def _difference_series(self, series, order=1):
        result = series.copy()
        for _ in range(order):
            result = np.diff(result)
        return result
    
    def _add_lag_features(self, X, y, n_lags):
        """Add lagged y values as features"""
        lagged_features = []
        
        for lag in range(1, n_lags + 1):
            lagged_y = np.roll(y, lag)
            lagged_y[:lag] = np.nan  # Set first 'lag' values to NaN
            lagged_features.append(lagged_y.reshape(-1, 1))
        
        if X is not None:
            return np.concatenate([X] + lagged_features, axis=1)
        else:
            return np.concatenate(lagged_features, axis=1)


def get_imputer(config):
    imp_cfg = config['preprocessing']['imputer']
    
    if imp_cfg['type'] == 'TimeAwareKNNImputer':
        imputer = TimeAwareKNNImputer(**imp_cfg['params'])
    else:
        raise ValueError(f"Onbekende imputer: {imp_cfg['type']}")
    return imputer

# Custom transformer for initial data preprocessing
class WeatherDataPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, freq='3h', fill_method='interpolate', add_time_dummies=None):
        """
        Preprocesses the weather data:
        - Converts categorical vars to numeric
        - Sets datetime index
        - Fills missing rows/timestamps
        - Adds cyclical dummies for hour, day of week, and month
        """
        self.freq = freq
        self.fill_method = fill_method
        self.add_time_dummies = add_time_dummies
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        df = X.copy()
        
        # Convert Valencia_wind_deg to numerical values if it exists
        if 'Valencia_wind_deg' in df.columns:
            df['Valencia_wind_deg_cat'] = df['Valencia_wind_deg'].astype(str).str.replace('level_', '').astype(float)
            df = df.drop(columns=['Valencia_wind_deg'])

        # Convert Seville_pressure to numerical values if it exists
        if 'Seville_pressure' in df.columns:
            df['Seville_pressure_cat'] = df['Seville_pressure'].astype(str).str.replace('sp', '').astype(float)
            df = df.drop(columns=['Seville_pressure'])
        
        # Ensure time is datetime type
        if 'time' in df.columns:
            df = self._set_datetime_as_index(df)
        
        # Remove any unwanted columns
        if 'Unnamed: 0' in df.columns:
            df = df.drop(columns=['Unnamed: 0'])

        # Add cyclical time features
        if self.add_time_dummies == "cyclical":
            df = self._add_cyclical_time_features(df)
        
        return df
    
    def _set_datetime_as_index(self, df):
        """
        Convert 'time' column to datetime index, reindex to regular intervals,
        and impute missing rows.
        """
        # Convert and sort time
        df['time'] = pd.to_datetime(df['time'])
        df = df.set_index('time')

        # Create complete datetime index
        full_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq=self.freq)

        # Reindex
        df = df.reindex(full_index)

        # Identify fully missing rows
        fully_missing_mask = df.isna().all(axis=1)
         
        """
        # Apply imputation only to fully missing rows
        if self.fill_method == 'interpolate':
            df.loc[fully_missing_mask] = df.interpolate().loc[fully_missing_mask]
        elif self.fill_method == 'ffill':
            df.loc[fully_missing_mask] = df.ffill().loc[fully_missing_mask]
        elif self.fill_method == 'bfill':
            df.loc[fully_missing_mask] = df.bfill().loc[fully_missing_mask]
        elif self.fill_method == 'zero':
            df.loc[fully_missing_mask] = 0
        else:
            raise ValueError("Unsupported fill_method. Choose from ['interpolate', 'ffill', 'bfill', 'zero'].")"""
        

        df = df.asfreq(self.freq)
        df.drop(index=df.index[fully_missing_mask], inplace=True)

        return df
    

    def _add_cyclical_time_features(self, df):
        # Extract time components
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['month'] = df.index.month

        # Add cyclical encoding
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

        df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

        return df


class ToNumpyArray(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X.to_numpy() if isinstance(X, pd.DataFrame) else X

# Imputation Methods

# 1. Hour/ day/ month median imputer

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

# 2. Time-based Method: Interpolation with Fallback
class InterpolationImputer(BaseEstimator, TransformerMixin):
    def __init__(self, column='Valencia_pressure', max_gap=24):
        """
        Time-based interpolation with fallback to pattern-based imputation
        
        Parameters:
        column (str): The column to impute
        max_gap (int): Maximum gap size (in hours) to interpolate across
        """
        self.column = column
        self.max_gap = max_gap
    
    def fit(self, X, y=None):
        # For fallback pattern-based imputation
        df = X.copy()
        df['hour'] = df.index.hour
        df['day'] = df.index.dayofweek
        
        # Using just hour and day of week for more robust patterns
        self.hourly_pattern_ = df.groupby('hour')[self.column].median()
        self.daily_pattern_ = df.groupby('day')[self.column].median()
        
        # Store global median as last resort
        self.global_median_ = df[self.column].median()
        
        return self
    
    def transform(self, X):
        df = X.copy()
        
        # First identify missing value locations
        missing_mask = df[self.column].isna()
        
        if missing_mask.sum() > 0:
            # Sort index to ensure time ordering
            df = df.sort_index()
            
            # Copy original series for reference
            original_series = df[self.column].copy()
            
            # Step 1: Perform interpolation for gaps smaller than max_gap
            # Create a helper series to identify gap sizes
            gap_helper = pd.Series(index=df.index, data=np.arange(len(df)))
            gap_helper[~missing_mask] = np.nan
            gap_helper = gap_helper.fillna(method='ffill')
            
            # Identify consecutive stretches of missing values
            stretches = gap_helper.dropna().diff().dropna()
            small_gaps = stretches[stretches <= self.max_gap].index
            
            # Perform interpolation only on small gaps
            if len(small_gaps) > 0:
                df_interp = df.copy()
                df_interp[self.column] = df_interp[self.column].interpolate(
                    method='time', limit_area='inside')
                
                # Apply interpolated values only to small gaps
                for idx in small_gaps:
                    start_idx = df.index.get_loc(idx) - int(stretches[idx])
                    end_idx = df.index.get_loc(idx) + 1
                    mask = slice(start_idx, end_idx)
                    df.loc[df.index[mask], self.column] = df_interp.loc[df.index[mask], self.column]
            
            # Step 2: Fall back to pattern-based imputation for remaining NaNs
            still_missing = df[self.column].isna()
            
            if still_missing.sum() > 0:
                df['hour'] = df.index.hour
                df['day'] = df.index.dayofweek
                
                # Apply hour-of-day pattern
                for idx in df[still_missing].index:
                    hour = idx.hour
                    day = idx.dayofweek
                    
                    if hour in self.hourly_pattern_:
                        df.loc[idx, self.column] = self.hourly_pattern_[hour]
                    elif day in self.daily_pattern_:
                        df.loc[idx, self.column] = self.daily_pattern_[day]
                    else:
                        df.loc[idx, self.column] = self.global_median_
                
                df.drop(['hour', 'day'], axis=1, inplace=True)
        
        return df


# 3. ML-based Method: KNN Imputation with Time Features
class TimeAwareKNNImputer(BaseEstimator, TransformerMixin):
    def __init__(self, column='Valencia_pressure', n_neighbors=5):
        """
        KNN-based imputation that incorporates time features
        
        Parameters:
        column (str): The column to impute
        n_neighbors (int): Number of neighbors to use for KNN imputation
        """
        self.column = column
        self.n_neighbors = n_neighbors
        self.imputer = KNNImputer(n_neighbors=n_neighbors)
    
    def fit(self, X, y=None):
        # No need to fit anything here as KNN imputation is lazy
        return self
    
    def transform(self, X):
        df = X.copy()
        
        # Skip if the column doesn't exist or has no missing values
        if self.column not in df.columns or not df[self.column].isna().any():
            return df
        
        # Create time-based features to help with imputation
        df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
        df['day_sin'] = np.sin(2 * np.pi * df.index.dayofweek / 7)
        df['day_cos'] = np.cos(2 * np.pi * df.index.dayofweek / 7)
        df['month_sin'] = np.sin(2 * np.pi * df.index.month / 12)
        df['month_cos'] = np.cos(2 * np.pi * df.index.month / 12)
        
        # Extract any other columns that might be useful for imputation
        feature_cols = ['hour_sin', 'hour_cos', 'day_sin', 'day_cos', 
                      'month_sin', 'month_cos', self.column]
        
        # Add any other available columns that might correlate with pressure
        for col in df.columns:
            if col not in feature_cols and col not in [self.column]:
                if df[col].dtype in [np.float64, np.int64]:
                    feature_cols.append(col)
        
        # Apply KNN imputation
        imputed_data = self.imputer.fit_transform(df[feature_cols])
        
        # Put imputed pressure back
        df[self.column] = imputed_data[:, feature_cols.index(self.column)]
        
        # Drop the added time features
        df.drop(['hour_sin', 'hour_cos', 'day_sin', 'day_cos', 
                'month_sin', 'month_cos'], axis=1, inplace=True)
        
        return df


# 4. Simplified Grouping Method: Hour and Day of Week
class SimplifiedPatternImputer(BaseEstimator, TransformerMixin):
    def __init__(self, column='Valencia_pressure'):
        """
        Pattern-based imputation using hierarchical grouping approach
        
        Parameters:
        column (str): The column to impute
        """
        self.column = column
    
    def fit(self, X, y=None):
        df = X.copy()
        df['hour'] = df.index.hour
        df['day'] = df.index.dayofweek
        
        # Create different levels of patterns from most specific to most general
        self.hour_day_medians_ = df.groupby(['hour', 'day'])[self.column].median()
        self.hour_medians_ = df.groupby(['hour'])[self.column].median()
        self.day_medians_ = df.groupby(['day'])[self.column].median()
        self.global_median_ = df[self.column].median()
        
        return self
    
    def transform(self, X):
        df = X.copy()
        
        # Skip if the column doesn't exist or has no missing values
        if self.column not in df.columns or not df[self.column].isna().any():
            return df
            
        df['hour'] = df.index.hour
        df['day'] = df.index.dayofweek
        
        def impute(row):
            if pd.isna(row[self.column]):
                # Try most specific pattern first
                pressure = self.hour_day_medians_.get((row['hour'], row['day']), np.nan)
                
                # Fall back to hour pattern
                if pd.isna(pressure):
                    pressure = self.hour_medians_.get(row['hour'], np.nan)
                
                # Fall back to day pattern
                if pd.isna(pressure):
                    pressure = self.day_medians_.get(row['day'], np.nan)
                    
                # Last resort: global median
                if pd.isna(pressure):
                    pressure = self.global_median_
                
                return pressure
            return row[self.column]
        
        df[self.column] = df.apply(impute, axis=1)
        df.drop(['hour', 'day'], axis=1, inplace=True)
        
        return df





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

    # Identify fully missing rows
    fully_missing_mask = df.isna().all(axis=1)

    # Apply imputation only to fully missing rows
    if fill_method == 'interpolate':
        df.loc[fully_missing_mask] = df.interpolate().loc[fully_missing_mask]
    elif fill_method == 'ffill':
        df.loc[fully_missing_mask] = df.ffill().loc[fully_missing_mask]
    elif fill_method == 'bfill':
        df.loc[fully_missing_mask] = df.bfill().loc[fully_missing_mask]
    elif fill_method == 'zero':
        df.loc[fully_missing_mask] = 0
    else:
        raise ValueError("Unsupported fill_method. Choose from ['interpolate', 'ffill', 'bfill', 'zero'].")

    df = df.asfreq('3h')

    return df



    
# 2. ML-based Method: KNN Imputation with Time Features
class TimeAwareKNNImputer(BaseEstimator, TransformerMixin):
    def __init__(self, n_neighbors=5):
        """
        KNN-based imputation that incorporates time features
        
        Parameters:
        n_neighbors (int): Number of neighbors to use for KNN imputation
        """
        self.n_neighbors = n_neighbors
        self.imputer = KNNImputer(n_neighbors=n_neighbors)
    
    def fit(self, X, y=None):
        # No need to fit anything here as KNN imputation is lazy
        return self
    
    def transform(self, X):
        df = X.copy()
        
        # Create time-based features to help with imputation
        df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
        df['day_sin'] = np.sin(2 * np.pi * df.index.dayofweek / 7)
        df['day_cos'] = np.cos(2 * np.pi * df.index.dayofweek / 7)
        df['month_sin'] = np.sin(2 * np.pi * df.index.month / 12)
        df['month_cos'] = np.cos(2 * np.pi * df.index.month / 12)
        
        # Extract any other columns that might be useful for imputation
        # Here we'll assume there might be temperature or other weather features
        feature_cols = ['hour_sin', 'hour_cos', 'day_sin', 'day_cos', 
                      'month_sin', 'month_cos', 'Valencia_pressure']
        
        # Add any other available columns that might correlate with pressure
        # Like temperature, humidity, etc.
        for col in df.columns:
            if col not in feature_cols and col != 'Valencia_pressure':
                if df[col].dtype in [np.float64, np.int64]:
                    feature_cols.append(col)
        
        # Apply KNN imputation
        imputed_data = self.imputer.fit_transform(df[feature_cols])
        
        # Put imputed pressure back
        df['Valencia_pressure'] = imputed_data[:, feature_cols.index('Valencia_pressure')]
        
        # Drop the added time features
        df.drop(['hour_sin', 'hour_cos', 'day_sin', 'day_cos', 
                'month_sin', 'month_cos'], axis=1, inplace=True)
        
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
