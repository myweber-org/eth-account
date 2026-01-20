
import numpy as np
import pandas as pd
from scipy import stats

def normalize_data(data, method='zscore'):
    """
    Normalize data using specified method.
    
    Args:
        data: numpy array or pandas Series
        method: 'zscore', 'minmax', or 'robust'
    
    Returns:
        Normalized data
    """
    if method == 'zscore':
        return (data - np.mean(data)) / np.std(data)
    elif method == 'minmax':
        return (data - np.min(data)) / (np.max(data) - np.min(data))
    elif method == 'robust':
        return (data - np.median(data)) / stats.iqr(data)
    else:
        raise ValueError("Method must be 'zscore', 'minmax', or 'robust'")

def remove_outliers_iqr(data, factor=1.5):
    """
    Remove outliers using IQR method.
    
    Args:
        data: numpy array or pandas Series
        factor: multiplier for IQR (default 1.5)
    
    Returns:
        Data with outliers removed
    """
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower_bound = q1 - factor * iqr
    upper_bound = q3 + factor * iqr
    
    return data[(data >= lower_bound) & (data <= upper_bound)]

def clean_dataset(df, columns=None, outlier_method='iqr', normalize_method='zscore'):
    """
    Clean dataset by removing outliers and normalizing specified columns.
    
    Args:
        df: pandas DataFrame
        columns: list of columns to clean (default: all numeric columns)
        outlier_method: 'iqr' or None
        normalize_method: 'zscore', 'minmax', 'robust', or None
    
    Returns:
        Cleaned DataFrame
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    cleaned_df = df.copy()
    
    for col in columns:
        if col in df.columns:
            # Remove outliers
            if outlier_method == 'iqr':
                mask = ~df[col].isna()
                clean_series = remove_outliers_iqr(df[col][mask])
                cleaned_df.loc[mask, col] = clean_series
            
            # Normalize data
            if normalize_method:
                mask = ~cleaned_df[col].isna()
                cleaned_df.loc[mask, col] = normalize_data(
                    cleaned_df.loc[mask, col], 
                    method=normalize_method
                )
    
    return cleaned_df

def calculate_statistics(data):
    """
    Calculate basic statistics for data.
    
    Args:
        data: numpy array or pandas Series
    
    Returns:
        Dictionary of statistics
    """
    return {
        'mean': np.mean(data),
        'median': np.median(data),
        'std': np.std(data),
        'min': np.min(data),
        'max': np.max(data),
        'q1': np.percentile(data, 25),
        'q3': np.percentile(data, 75),
        'skewness': stats.skew(data),
        'kurtosis': stats.kurtosis(data)
    }