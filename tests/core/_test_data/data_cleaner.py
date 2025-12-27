
import numpy as np
import pandas as pd

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to process
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
    return filtered_df

def calculate_summary_stats(df, column):
    """
    Calculate summary statistics for a column.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name
    
    Returns:
    dict: Dictionary containing summary statistics
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    stats = {
        'mean': df[column].mean(),
        'median': df[column].median(),
        'std': df[column].std(),
        'min': df[column].min(),
        'max': df[column].max(),
        'count': df[column].count(),
        'missing': df[column].isnull().sum()
    }
    
    return stats

def normalize_column(df, column, method='minmax'):
    """
    Normalize a column using specified method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to normalize
    method (str): Normalization method ('minmax' or 'zscore')
    
    Returns:
    pd.DataFrame: DataFrame with normalized column
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    df_copy = df.copy()
    
    if method == 'minmax':
        min_val = df_copy[column].min()
        max_val = df_copy[column].max()
        if max_val != min_val:
            df_copy[f'{column}_normalized'] = (df_copy[column] - min_val) / (max_val - min_val)
        else:
            df_copy[f'{column}_normalized'] = 0.5
    
    elif method == 'zscore':
        mean_val = df_copy[column].mean()
        std_val = df_copy[column].std()
        if std_val > 0:
            df_copy[f'{column}_normalized'] = (df_copy[column] - mean_val) / std_val
        else:
            df_copy[f'{column}_normalized'] = 0
    
    else:
        raise ValueError("Method must be 'minmax' or 'zscore'")
    
    return df_copy

def handle_missing_values(df, column, strategy='mean'):
    """
    Handle missing values in a column.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name
    strategy (str): Strategy to handle missing values ('mean', 'median', 'mode', 'drop')
    
    Returns:
    pd.DataFrame: DataFrame with handled missing values
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    df_copy = df.copy()
    
    if strategy == 'mean':
        fill_value = df_copy[column].mean()
    elif strategy == 'median':
        fill_value = df_copy[column].median()
    elif strategy == 'mode':
        fill_value = df_copy[column].mode()[0] if not df_copy[column].mode().empty else 0
    elif strategy == 'drop':
        df_copy = df_copy.dropna(subset=[column])
        return df_copy
    else:
        raise ValueError("Strategy must be 'mean', 'median', 'mode', or 'drop'")
    
    df_copy[column] = df_copy[column].fillna(fill_value)
    return df_copyimport pandas as pd
import numpy as np
from typing import List, Union

def remove_duplicates(df: pd.DataFrame, subset: List[str] = None) -> pd.DataFrame:
    """
    Remove duplicate rows from DataFrame.
    
    Args:
        df: Input DataFrame
        subset: Columns to consider for identifying duplicates
    
    Returns:
        DataFrame with duplicates removed
    """
    return df.drop_duplicates(subset=subset, keep='first')

def convert_column_types(df: pd.DataFrame, 
                         column_types: dict) -> pd.DataFrame:
    """
    Convert columns to specified data types.
    
    Args:
        df: Input DataFrame
        column_types: Dictionary mapping column names to target types
    
    Returns:
        DataFrame with converted column types
    """
    df_copy = df.copy()
    for column, dtype in column_types.items():
        if column in df_copy.columns:
            try:
                df_copy[column] = df_copy[column].astype(dtype)
            except (ValueError, TypeError):
                df_copy[column] = pd.to_numeric(df_copy[column], errors='coerce')
    return df_copy

def handle_missing_values(df: pd.DataFrame, 
                          strategy: str = 'mean',
                          columns: List[str] = None) -> pd.DataFrame:
    """
    Handle missing values in DataFrame.
    
    Args:
        df: Input DataFrame
        strategy: Imputation strategy ('mean', 'median', 'mode', 'drop')
        columns: Specific columns to process
    
    Returns:
        DataFrame with handled missing values
    """
    df_copy = df.copy()
    
    if columns is None:
        columns = df_copy.columns
    
    for column in columns:
        if column in df_copy.columns and df_copy[column].isnull().any():
            if strategy == 'mean':
                df_copy[column].fillna(df_copy[column].mean(), inplace=True)
            elif strategy == 'median':
                df_copy[column].fillna(df_copy[column].median(), inplace=True)
            elif strategy == 'mode':
                df_copy[column].fillna(df_copy[column].mode()[0], inplace=True)
            elif strategy == 'drop':
                df_copy = df_copy.dropna(subset=[column])
    
    return df_copy

def normalize_numeric_columns(df: pd.DataFrame,
                             columns: List[str] = None,
                             method: str = 'minmax') -> pd.DataFrame:
    """
    Normalize numeric columns in DataFrame.
    
    Args:
        df: Input DataFrame
        columns: Columns to normalize
        method: Normalization method ('minmax' or 'zscore')
    
    Returns:
        DataFrame with normalized columns
    """
    df_copy = df.copy()
    
    if columns is None:
        columns = df_copy.select_dtypes(include=[np.number]).columns
    
    for column in columns:
        if column in df_copy.columns and pd.api.types.is_numeric_dtype(df_copy[column]):
            if method == 'minmax':
                min_val = df_copy[column].min()
                max_val = df_copy[column].max()
                if max_val > min_val:
                    df_copy[column] = (df_copy[column] - min_val) / (max_val - min_val)
            elif method == 'zscore':
                mean_val = df_copy[column].mean()
                std_val = df_copy[column].std()
                if std_val > 0:
                    df_copy[column] = (df_copy[column] - mean_val) / std_val
    
    return df_copy

def clean_dataframe(df: pd.DataFrame,
                   remove_dups: bool = True,
                   type_conversions: dict = None,
                   missing_strategy: str = 'mean',
                   normalize: bool = False,
                   normalize_method: str = 'minmax') -> pd.DataFrame:
    """
    Comprehensive data cleaning pipeline.
    
    Args:
        df: Input DataFrame
        remove_dups: Whether to remove duplicates
        type_conversions: Dictionary of column type conversions
        missing_strategy: Strategy for handling missing values
        normalize: Whether to normalize numeric columns
        normalize_method: Normalization method
    
    Returns:
        Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    if remove_dups:
        cleaned_df = remove_duplicates(cleaned_df)
    
    if type_conversions:
        cleaned_df = convert_column_types(cleaned_df, type_conversions)
    
    if missing_strategy != 'none':
        cleaned_df = handle_missing_values(cleaned_df, strategy=missing_strategy)
    
    if normalize:
        cleaned_df = normalize_numeric_columns(cleaned_df, method=normalize_method)
    
    return cleaned_df