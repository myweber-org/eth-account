
import pandas as pd
import numpy as np

def remove_missing_rows(df, columns=None):
    """
    Remove rows with missing values from specified columns or entire DataFrame.
    
    Args:
        df: pandas DataFrame
        columns: list of column names or None for all columns
    
    Returns:
        Cleaned DataFrame
    """
    if columns:
        return df.dropna(subset=columns)
    return df.dropna()

def fill_missing_values(df, strategy='mean', columns=None):
    """
    Fill missing values using specified strategy.
    
    Args:
        df: pandas DataFrame
        strategy: 'mean', 'median', 'mode', or 'constant'
        columns: list of column names or None for all columns
    
    Returns:
        DataFrame with filled values
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

def detect_outliers_iqr(df, column, multiplier=1.5):
    """
    Detect outliers using Interquartile Range method.
    
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

def remove_outliers(df, columns=None, multiplier=1.5):
    """
    Remove outliers from specified columns using IQR method.
    
    Args:
        df: pandas DataFrame
        columns: list of column names or None for all numeric columns
        multiplier: IQR multiplier
    
    Returns:
        DataFrame without outliers
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    mask = pd.Series([False] * len(df))
    
    for col in columns:
        if df[col].dtype in ['int64', 'float64']:
            outliers = detect_outliers_iqr(df, col, multiplier)
            mask = mask | outliers
    
    return df[~mask]

def normalize_column(df, column, method='minmax'):
    """
    Normalize a column using specified method.
    
    Args:
        df: pandas DataFrame
        column: column name to normalize
        method: 'minmax' or 'zscore'
    
    Returns:
        Series with normalized values
    """
    if method == 'minmax':
        min_val = df[column].min()
        max_val = df[column].max()
        if max_val == min_val:
            return pd.Series([0.5] * len(df))
        return (df[column] - min_val) / (max_val - min_val)
    
    elif method == 'zscore':
        mean_val = df[column].mean()
        std_val = df[column].std()
        if std_val == 0:
            return pd.Series([0] * len(df))
        return (df[column] - mean_val) / std_val
    
    return df[column].copy()

def clean_dataset(df, missing_strategy='mean', remove_outliers_flag=True):
    """
    Comprehensive dataset cleaning pipeline.
    
    Args:
        df: pandas DataFrame
        missing_strategy: strategy for handling missing values
        remove_outliers_flag: whether to remove outliers
    
    Returns:
        Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
    
    cleaned_df = fill_missing_values(cleaned_df, strategy=missing_strategy)
    
    if remove_outliers_flag and len(numeric_cols) > 0:
        cleaned_df = remove_outliers(cleaned_df, columns=numeric_cols)
    
    return cleaned_df