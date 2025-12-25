
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(df, columns=None, threshold=1.5):
    """
    Remove outliers using IQR method
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    df_clean = df.copy()
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        
        mask = (df[col] >= lower_bound) & (df[col] <= upper_bound)
        df_clean = df_clean[mask]
    
    return df_clean.reset_index(drop=True)

def remove_outliers_zscore(df, columns=None, threshold=3):
    """
    Remove outliers using Z-score method
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    df_clean = df.copy()
    for col in columns:
        z_scores = np.abs(stats.zscore(df[col].dropna()))
        mask = z_scores < threshold
        df_clean = df_clean[mask]
    
    return df_clean.reset_index(drop=True)

def normalize_minmax(df, columns=None):
    """
    Normalize data using Min-Max scaling
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    df_normalized = df.copy()
    for col in columns:
        min_val = df[col].min()
        max_val = df[col].max()
        if max_val != min_val:
            df_normalized[col] = (df[col] - min_val) / (max_val - min_val)
    
    return df_normalized

def normalize_zscore(df, columns=None):
    """
    Normalize data using Z-score standardization
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    df_normalized = df.copy()
    for col in columns:
        mean_val = df[col].mean()
        std_val = df[col].std()
        if std_val != 0:
            df_normalized[col] = (df[col] - mean_val) / std_val
    
    return df_normalized

def handle_missing_values(df, strategy='mean', columns=None):
    """
    Handle missing values with different strategies
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    df_filled = df.copy()
    
    for col in columns:
        if strategy == 'mean':
            fill_value = df[col].mean()
        elif strategy == 'median':
            fill_value = df[col].median()
        elif strategy == 'mode':
            fill_value = df[col].mode()[0]
        elif strategy == 'zero':
            fill_value = 0
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        df_filled[col] = df[col].fillna(fill_value)
    
    return df_filled

def clean_data_pipeline(df, outlier_method='iqr', normalize_method='minmax', 
                       missing_strategy='mean', outlier_threshold=1.5):
    """
    Complete data cleaning pipeline
    """
    # Handle missing values first
    df_clean = handle_missing_values(df, strategy=missing_strategy)
    
    # Remove outliers
    if outlier_method == 'iqr':
        df_clean = remove_outliers_iqr(df_clean, threshold=outlier_threshold)
    elif outlier_method == 'zscore':
        df_clean = remove_outliers_zscore(df_clean, threshold=outlier_threshold)
    
    # Normalize data
    if normalize_method == 'minmax':
        df_clean = normalize_minmax(df_clean)
    elif normalize_method == 'zscore':
        df_clean = normalize_zscore(df_clean)
    
    return df_clean