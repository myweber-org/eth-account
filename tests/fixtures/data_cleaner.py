import pandas as pd
import numpy as np

def remove_duplicates(df, subset=None):
    """
    Remove duplicate rows from DataFrame.
    
    Args:
        df: pandas DataFrame
        subset: column label or sequence of labels to consider for duplicates
    
    Returns:
        DataFrame with duplicates removed
    """
    return df.drop_duplicates(subset=subset, keep='first')

def fill_missing_values(df, strategy='mean', columns=None):
    """
    Fill missing values in DataFrame.
    
    Args:
        df: pandas DataFrame
        strategy: 'mean', 'median', 'mode', or 'constant'
        columns: list of columns to fill, None for all columns
    
    Returns:
        DataFrame with missing values filled
    """
    df_filled = df.copy()
    
    if columns is None:
        columns = df.columns
    
    for col in columns:
        if df[col].dtype in ['int64', 'float64']:
            if strategy == 'mean':
                df_filled[col] = df[col].fillna(df[col].mean())
            elif strategy == 'median':
                df_filled[col] = df[col].fillna(df[col].median())
            elif strategy == 'mode':
                df_filled[col] = df[col].fillna(df[col].mode()[0])
            elif strategy == 'constant':
                df_filled[col] = df[col].fillna(0)
        else:
            df_filled[col] = df[col].fillna(df[col].mode()[0])
    
    return df_filled

def remove_outliers(df, columns=None, method='iqr', threshold=1.5):
    """
    Remove outliers from DataFrame using specified method.
    
    Args:
        df: pandas DataFrame
        columns: list of numeric columns to check for outliers
        method: 'iqr' for interquartile range or 'zscore' for standard deviation
        threshold: multiplier for IQR or cutoff for z-score
    
    Returns:
        DataFrame with outliers removed
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    df_clean = df.copy()
    
    for col in columns:
        if method == 'iqr':
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            mask = (df[col] >= lower_bound) & (df[col] <= upper_bound)
        elif method == 'zscore':
            z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
            mask = z_scores < threshold
        
        df_clean = df_clean[mask]
    
    return df_clean.reset_index(drop=True)

def normalize_data(df, columns=None, method='minmax'):
    """
    Normalize numeric columns in DataFrame.
    
    Args:
        df: pandas DataFrame
        columns: list of columns to normalize
        method: 'minmax' or 'standard'
    
    Returns:
        DataFrame with normalized columns
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    df_normalized = df.copy()
    
    for col in columns:
        if method == 'minmax':
            min_val = df[col].min()
            max_val = df[col].max()
            if max_val != min_val:
                df_normalized[col] = (df[col] - min_val) / (max_val - min_val)
        elif method == 'standard':
            mean_val = df[col].mean()
            std_val = df[col].std()
            if std_val != 0:
                df_normalized[col] = (df[col] - mean_val) / std_val
    
    return df_normalized

def clean_dataset(df, operations=None):
    """
    Apply multiple cleaning operations to DataFrame.
    
    Args:
        df: pandas DataFrame
        operations: dictionary of operations and parameters
    
    Returns:
        Cleaned DataFrame
    """
    if operations is None:
        operations = {
            'remove_duplicates': True,
            'fill_missing': {'strategy': 'mean'},
            'remove_outliers': {'method': 'iqr', 'threshold': 1.5}
        }
    
    df_clean = df.copy()
    
    if operations.get('remove_duplicates', False):
        df_clean = remove_duplicates(df_clean)
    
    if 'fill_missing' in operations:
        params = operations['fill_missing']
        df_clean = fill_missing_values(df_clean, **params)
    
    if 'remove_outliers' in operations:
        params = operations['remove_outliers']
        df_clean = remove_outliers(df_clean, **params)
    
    if 'normalize' in operations:
        params = operations['normalize']
        df_clean = normalize_data(df_clean, **params)
    
    return df_clean