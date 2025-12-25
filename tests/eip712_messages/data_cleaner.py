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
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        
        df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
    
    return df_clean

def normalize_minmax(df, columns=None):
    """
    Normalize data using min-max scaling
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    df_normalized = df.copy()
    for col in columns:
        min_val = df_normalized[col].min()
        max_val = df_normalized[col].max()
        if max_val != min_val:
            df_normalized[col] = (df_normalized[col] - min_val) / (max_val - min_val)
    
    return df_normalized

def handle_missing_values(df, strategy='mean', columns=None):
    """
    Handle missing values with specified strategy
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    df_filled = df.copy()
    for col in columns:
        if df_filled[col].isnull().any():
            if strategy == 'mean':
                fill_value = df_filled[col].mean()
            elif strategy == 'median':
                fill_value = df_filled[col].median()
            elif strategy == 'mode':
                fill_value = df_filled[col].mode()[0]
            else:
                fill_value = 0
            
            df_filled[col] = df_filled[col].fillna(fill_value)
    
    return df_filled

def clean_dataset(df, outlier_threshold=1.5, normalize=True, missing_strategy='mean'):
    """
    Complete data cleaning pipeline
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    df_processed = handle_missing_values(df, strategy=missing_strategy, columns=numeric_cols)
    df_processed = remove_outliers_iqr(df_processed, columns=numeric_cols, threshold=outlier_threshold)
    
    if normalize:
        df_processed = normalize_minmax(df_processed, columns=numeric_cols)
    
    return df_processed