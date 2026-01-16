
import pandas as pd
import numpy as np
from scipy import stats

def normalize_data(df, columns=None, method='zscore'):
    """
    Normalize specified columns in DataFrame.
    
    Args:
        df: pandas DataFrame
        columns: list of column names to normalize (default: all numeric columns)
        method: 'zscore', 'minmax', or 'robust'
    
    Returns:
        Normalized DataFrame
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    df_normalized = df.copy()
    
    for col in columns:
        if col not in df.columns:
            continue
            
        if method == 'zscore':
            df_normalized[col] = (df[col] - df[col].mean()) / df[col].std()
        elif method == 'minmax':
            df_normalized[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
        elif method == 'robust':
            df_normalized[col] = (df[col] - df[col].median()) / stats.iqr(df[col])
    
    return df_normalized

def remove_outliers(df, columns=None, method='iqr', threshold=1.5):
    """
    Remove outliers from specified columns.
    
    Args:
        df: pandas DataFrame
        columns: list of column names to check for outliers
        method: 'iqr' or 'zscore'
        threshold: multiplier for IQR or cutoff for z-score
    
    Returns:
        DataFrame with outliers removed
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    df_clean = df.copy()
    
    for col in columns:
        if col not in df.columns:
            continue
            
        if method == 'iqr':
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            mask = (df[col] >= lower_bound) & (df[col] <= upper_bound)
        elif method == 'zscore':
            z_scores = np.abs(stats.zscore(df[col].fillna(df[col].mean())))
            mask = z_scores < threshold
        
        df_clean = df_clean[mask]
    
    return df_clean.reset_index(drop=True)

def handle_missing_values(df, strategy='mean', columns=None):
    """
    Handle missing values in DataFrame.
    
    Args:
        df: pandas DataFrame
        strategy: 'mean', 'median', 'mode', or 'drop'
        columns: list of column names to process
    
    Returns:
        DataFrame with handled missing values
    """
    if columns is None:
        columns = df.columns
    
    df_processed = df.copy()
    
    for col in columns:
        if col not in df.columns:
            continue
            
        if strategy == 'mean':
            df_processed[col].fillna(df[col].mean(), inplace=True)
        elif strategy == 'median':
            df_processed[col].fillna(df[col].median(), inplace=True)
        elif strategy == 'mode':
            df_processed[col].fillna(df[col].mode()[0], inplace=True)
        elif strategy == 'drop':
            df_processed = df_processed.dropna(subset=[col])
    
    return df_processed

def validate_data(df, checks=None):
    """
    Validate data quality with various checks.
    
    Args:
        df: pandas DataFrame
        checks: list of checks to perform
    
    Returns:
        Dictionary with validation results
    """
    if checks is None:
        checks = ['missing', 'duplicates', 'negative_values']
    
    results = {}
    
    if 'missing' in checks:
        results['missing_values'] = df.isnull().sum().to_dict()
    
    if 'duplicates' in checks:
        results['duplicate_rows'] = df.duplicated().sum()
    
    if 'negative_values' in checks:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        negative_counts = {}
        for col in numeric_cols:
            negative_counts[col] = (df[col] < 0).sum()
        results['negative_values'] = negative_counts
    
    return results