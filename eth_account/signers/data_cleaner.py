
import pandas as pd
import numpy as np
from scipy import stats

def detect_outliers_iqr(data_series, threshold=1.5):
    """
    Detect outliers using Interquartile Range method.
    Returns boolean mask where True indicates an outlier.
    """
    q1 = data_series.quantile(0.25)
    q3 = data_series.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr
    return (data_series < lower_bound) | (data_series > upper_bound)

def remove_outliers(df, column_names, method='iqr', **kwargs):
    """
    Remove outliers from specified columns in DataFrame.
    Supports 'iqr' and 'zscore' methods.
    """
    df_clean = df.copy()
    
    for col in column_names:
        if col not in df_clean.columns:
            continue
            
        if method == 'iqr':
            threshold = kwargs.get('threshold', 1.5)
            outlier_mask = detect_outliers_iqr(df_clean[col], threshold)
        elif method == 'zscore':
            zscore_threshold = kwargs.get('zscore_threshold', 3)
            z_scores = np.abs(stats.zscore(df_clean[col].dropna()))
            outlier_mask = z_scores > zscore_threshold
            outlier_mask = pd.Series(outlier_mask, index=df_clean[col].dropna().index)
        else:
            raise ValueError(f"Unsupported method: {method}")
        
        df_clean.loc[outlier_mask, col] = np.nan
    
    return df_clean

def normalize_data(df, column_names, method='minmax'):
    """
    Normalize specified columns using different scaling methods.
    """
    df_normalized = df.copy()
    
    for col in column_names:
        if col not in df_normalized.columns:
            continue
            
        if method == 'minmax':
            col_min = df_normalized[col].min()
            col_max = df_normalized[col].max()
            if col_max != col_min:
                df_normalized[col] = (df_normalized[col] - col_min) / (col_max - col_min)
        
        elif method == 'standard':
            col_mean = df_normalized[col].mean()
            col_std = df_normalized[col].std()
            if col_std != 0:
                df_normalized[col] = (df_normalized[col] - col_mean) / col_std
        
        elif method == 'robust':
            col_median = df_normalized[col].median()
            col_iqr = df_normalized[col].quantile(0.75) - df_normalized[col].quantile(0.25)
            if col_iqr != 0:
                df_normalized[col] = (df_normalized[col] - col_median) / col_iqr
    
    return df_normalized

def handle_missing_values(df, strategy='mean', fill_value=None):
    """
    Handle missing values in DataFrame.
    """
    df_filled = df.copy()
    
    numeric_cols = df_filled.select_dtypes(include=[np.number]).columns
    
    if strategy == 'mean':
        for col in numeric_cols:
            df_filled[col].fillna(df_filled[col].mean(), inplace=True)
    
    elif strategy == 'median':
        for col in numeric_cols:
            df_filled[col].fillna(df_filled[col].median(), inplace=True)
    
    elif strategy == 'mode':
        for col in numeric_cols:
            df_filled[col].fillna(df_filled[col].mode()[0], inplace=True)
    
    elif strategy == 'constant' and fill_value is not None:
        df_filled.fillna(fill_value, inplace=True)
    
    elif strategy == 'drop':
        df_filled.dropna(inplace=True)
    
    return df_filled

def clean_dataset(df, numeric_columns=None, outlier_method='iqr', 
                  normalize_method='standard', missing_strategy='mean'):
    """
    Complete data cleaning pipeline.
    """
    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    df_clean = df.copy()
    
    df_clean = remove_outliers(df_clean, numeric_columns, method=outlier_method)
    df_clean = handle_missing_values(df_clean, strategy=missing_strategy)
    df_clean = normalize_data(df_clean, numeric_columns, method=normalize_method)
    
    return df_clean