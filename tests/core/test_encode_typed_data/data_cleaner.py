
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(dataframe, column, threshold=1.5):
    """
    Remove outliers from a DataFrame column using IQR method.
    
    Args:
        dataframe: pandas DataFrame
        column: column name to process
        threshold: IQR multiplier (default 1.5)
    
    Returns:
        DataFrame with outliers removed
    """
    if column not in dataframe.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    q1 = dataframe[column].quantile(0.25)
    q3 = dataframe[column].quantile(0.75)
    iqr = q3 - q1
    
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr
    
    filtered_df = dataframe[(dataframe[column] >= lower_bound) & 
                           (dataframe[column] <= upper_bound)]
    
    return filtered_df

def normalize_column_zscore(dataframe, column):
    """
    Normalize a column using Z-score normalization.
    
    Args:
        dataframe: pandas DataFrame
        column: column name to normalize
    
    Returns:
        DataFrame with normalized column added as '{column}_normalized'
    """
    if column not in dataframe.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    normalized_col = f"{column}_normalized"
    dataframe[normalized_col] = stats.zscore(dataframe[column])
    
    return dataframe

def min_max_normalize(dataframe, column, feature_range=(0, 1)):
    """
    Normalize a column using Min-Max scaling.
    
    Args:
        dataframe: pandas DataFrame
        column: column name to normalize
        feature_range: tuple of (min, max) for output range
    
    Returns:
        DataFrame with normalized column added as '{column}_scaled'
    """
    if column not in dataframe.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    min_val = dataframe[column].min()
    max_val = dataframe[column].max()
    
    scaled_col = f"{column}_scaled"
    dataframe[scaled_col] = ((dataframe[column] - min_val) / 
                            (max_val - min_val)) * (feature_range[1] - feature_range[0]) + feature_range[0]
    
    return dataframe

def clean_dataset(dataframe, numeric_columns, outlier_threshold=1.5, normalize_method='zscore'):
    """
    Comprehensive data cleaning pipeline.
    
    Args:
        dataframe: pandas DataFrame
        numeric_columns: list of numeric column names to process
        outlier_threshold: IQR threshold for outlier removal
        normalize_method: 'zscore' or 'minmax' normalization method
    
    Returns:
        Cleaned DataFrame
    """
    cleaned_df = dataframe.copy()
    
    for column in numeric_columns:
        if column in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, column, outlier_threshold)
            
            if normalize_method == 'zscore':
                cleaned_df = normalize_column_zscore(cleaned_df, column)
            elif normalize_method == 'minmax':
                cleaned_df = min_max_normalize(cleaned_df, column)
            else:
                raise ValueError("normalize_method must be 'zscore' or 'minmax'")
    
    return cleaned_df

def validate_dataframe(dataframe, required_columns=None, allow_nan=False):
    """
    Validate DataFrame structure and content.
    
    Args:
        dataframe: pandas DataFrame to validate
        required_columns: list of required column names
        allow_nan: whether to allow NaN values
    
    Returns:
        tuple of (is_valid, error_message)
    """
    if not isinstance(dataframe, pd.DataFrame):
        return False, "Input is not a pandas DataFrame"
    
    if dataframe.empty:
        return False, "DataFrame is empty"
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in dataframe.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
    
    if not allow_nan and dataframe.isnull().any().any():
        nan_columns = dataframe.columns[dataframe.isnull().any()].tolist()
        return False, f"NaN values found in columns: {nan_columns}"
    
    return True, "DataFrame is valid"