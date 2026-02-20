
import pandas as pd
import numpy as np

def clean_missing_values(df, strategy='mean', columns=None):
    """
    Handle missing values in a DataFrame using specified strategy.
    
    Args:
        df: pandas DataFrame containing data with potential missing values
        strategy: Method for handling missing values ('mean', 'median', 'mode', 'drop')
        columns: List of columns to apply cleaning to (None applies to all columns)
    
    Returns:
        Cleaned DataFrame with missing values handled
    """
    if df.empty:
        return df
    
    if columns is None:
        columns = df.columns
    
    df_clean = df.copy()
    
    for col in columns:
        if col not in df_clean.columns:
            continue
            
        if df_clean[col].isnull().sum() > 0:
            if strategy == 'mean':
                if pd.api.types.is_numeric_dtype(df_clean[col]):
                    df_clean[col].fillna(df_clean[col].mean(), inplace=True)
            elif strategy == 'median':
                if pd.api.types.is_numeric_dtype(df_clean[col]):
                    df_clean[col].fillna(df_clean[col].median(), inplace=True)
            elif strategy == 'mode':
                df_clean[col].fillna(df_clean[col].mode()[0] if not df_clean[col].mode().empty else np.nan, inplace=True)
            elif strategy == 'drop':
                df_clean = df_clean.dropna(subset=[col])
    
    return df_clean

def detect_outliers_iqr(df, column, multiplier=1.5):
    """
    Detect outliers using Interquartile Range method.
    
    Args:
        df: pandas DataFrame
        column: Column name to check for outliers
        multiplier: IQR multiplier for outlier detection
    
    Returns:
        Boolean Series indicating outlier rows
    """
    if column not in df.columns or not pd.api.types.is_numeric_dtype(df[column]):
        return pd.Series([False] * len(df), index=df.index)
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    return (df[column] < lower_bound) | (df[column] > upper_bound)

def normalize_column(df, column, method='minmax'):
    """
    Normalize a column using specified method.
    
    Args:
        df: pandas DataFrame
        column: Column name to normalize
        method: Normalization method ('minmax' or 'zscore')
    
    Returns:
        Series with normalized values
    """
    if column not in df.columns or not pd.api.types.is_numeric_dtype(df[column]):
        return df[column]
    
    if method == 'minmax':
        col_min = df[column].min()
        col_max = df[column].max()
        if col_max == col_min:
            return pd.Series([0.5] * len(df), index=df.index)
        return (df[column] - col_min) / (col_max - col_min)
    
    elif method == 'zscore':
        col_mean = df[column].mean()
        col_std = df[column].std()
        if col_std == 0:
            return pd.Series([0] * len(df), index=df.index)
        return (df[column] - col_mean) / col_std
    
    return df[column]