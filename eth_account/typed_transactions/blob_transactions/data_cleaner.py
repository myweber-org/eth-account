
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(df, column, multiplier=1.5):
    """
    Remove outliers from a DataFrame column using the IQR method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to process
    multiplier (float): IQR multiplier for outlier detection
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    
    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr
    
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
    return filtered_df.copy()

def remove_outliers_zscore(df, column, threshold=3):
    """
    Remove outliers using Z-score method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to process
    threshold (float): Z-score threshold for outlier detection
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    z_scores = np.abs(stats.zscore(df[column].dropna()))
    filtered_indices = np.where(z_scores < threshold)[0]
    
    filtered_df = df.iloc[filtered_indices].copy()
    
    return filtered_df

def clean_numeric_data(df, columns=None, method='iqr', **kwargs):
    """
    Clean numeric data by removing outliers from specified columns.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    columns (list): List of column names to clean. If None, clean all numeric columns.
    method (str): Outlier removal method ('iqr' or 'zscore')
    **kwargs: Additional arguments passed to outlier removal function
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_df = df.copy()
    
    for column in columns:
        if column in df.columns and pd.api.types.is_numeric_dtype(df[column]):
            if method == 'iqr':
                cleaned_df = remove_outliers_iqr(cleaned_df, column, **kwargs)
            elif method == 'zscore':
                cleaned_df = remove_outliers_zscore(cleaned_df, column, **kwargs)
            else:
                raise ValueError(f"Unknown method: {method}. Use 'iqr' or 'zscore'.")
    
    return cleaned_df.reset_index(drop=True)

def get_outlier_statistics(df, column):
    """
    Calculate outlier statistics for a column.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to analyze
    
    Returns:
    dict: Dictionary containing outlier statistics
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    
    stats_dict = {
        'total_count': len(df),
        'outlier_count': len(outliers),
        'outlier_percentage': (len(outliers) / len(df)) * 100,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'min_value': df[column].min(),
        'max_value': df[column].max(),
        'mean': df[column].mean(),
        'median': df[column].median()
    }
    
    return stats_dict