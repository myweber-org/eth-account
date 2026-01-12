
import pandas as pd
import numpy as np

def remove_missing_rows(df, columns=None):
    """
    Remove rows with missing values from DataFrame.
    
    Args:
        df: pandas DataFrame
        columns: list of columns to check for missing values
                 If None, checks all columns
    
    Returns:
        Cleaned DataFrame
    """
    if columns is None:
        columns = df.columns
    
    return df.dropna(subset=columns)

def replace_missing_with_mean(df, columns):
    """
    Replace missing values with column mean.
    
    Args:
        df: pandas DataFrame
        columns: list of columns to process
    
    Returns:
        DataFrame with missing values replaced
    """
    df_copy = df.copy()
    
    for col in columns:
        if col in df_copy.columns:
            mean_val = df_copy[col].mean()
            df_copy[col] = df_copy[col].fillna(mean_val)
    
    return df_copy

def detect_outliers_iqr(df, column, multiplier=1.5):
    """
    Detect outliers using IQR method.
    
    Args:
        df: pandas DataFrame
        column: column name to check for outliers
        multiplier: IQR multiplier (default 1.5)
    
    Returns:
        Boolean Series indicating outliers
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    return (df[column] < lower_bound) | (df[column] > upper_bound)

def cap_outliers(df, column, method='iqr', multiplier=1.5):
    """
    Cap outliers to specified bounds.
    
    Args:
        df: pandas DataFrame
        column: column name to process
        method: 'iqr' or 'percentile'
        multiplier: IQR multiplier (for 'iqr' method)
    
    Returns:
        DataFrame with capped outliers
    """
    df_copy = df.copy()
    
    if method == 'iqr':
        Q1 = df_copy[column].quantile(0.25)
        Q3 = df_copy[column].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        
    elif method == 'percentile':
        lower_bound = df_copy[column].quantile(0.01)
        upper_bound = df_copy[column].quantile(0.99)
    
    else:
        raise ValueError("Method must be 'iqr' or 'percentile'")
    
    df_copy[column] = df_copy[column].clip(lower=lower_bound, upper=upper_bound)
    
    return df_copy

def standardize_columns(df, columns):
    """
    Standardize columns to have mean=0 and std=1.
    
    Args:
        df: pandas DataFrame
        columns: list of columns to standardize
    
    Returns:
        DataFrame with standardized columns
    """
    df_copy = df.copy()
    
    for col in columns:
        if col in df_copy.columns:
            mean_val = df_copy[col].mean()
            std_val = df_copy[col].std()
            
            if std_val > 0:
                df_copy[col] = (df_copy[col] - mean_val) / std_val
    
    return df_copy

def clean_dataset(df, missing_strategy='remove', outlier_strategy='cap'):
    """
    Comprehensive data cleaning pipeline.
    
    Args:
        df: pandas DataFrame
        missing_strategy: 'remove' or 'mean'
        outlier_strategy: 'cap' or 'ignore'
    
    Returns:
        Cleaned DataFrame
    """
    df_clean = df.copy()
    
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
    
    if missing_strategy == 'remove':
        df_clean = remove_missing_rows(df_clean, numeric_cols)
    elif missing_strategy == 'mean':
        df_clean = replace_missing_with_mean(df_clean, numeric_cols)
    
    if outlier_strategy == 'cap':
        for col in numeric_cols:
            df_clean = cap_outliers(df_clean, col, method='iqr')
    
    df_clean = standardize_columns(df_clean, numeric_cols)
    
    return df_clean