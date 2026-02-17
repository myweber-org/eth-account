
import pandas as pd
import numpy as np
from typing import List, Union

def remove_duplicates(df: pd.DataFrame, subset: List[str] = None) -> pd.DataFrame:
    """
    Remove duplicate rows from DataFrame.
    
    Args:
        df: Input DataFrame
        subset: Columns to consider for duplicates
    
    Returns:
        DataFrame with duplicates removed
    """
    return df.drop_duplicates(subset=subset, keep='first')

def convert_column_types(df: pd.DataFrame, 
                         column_type_map: dict) -> pd.DataFrame:
    """
    Convert columns to specified data types.
    
    Args:
        df: Input DataFrame
        column_type_map: Dictionary mapping column names to target types
    
    Returns:
        DataFrame with converted column types
    """
    df_copy = df.copy()
    for column, dtype in column_type_map.items():
        if column in df_copy.columns:
            try:
                df_copy[column] = df_copy[column].astype(dtype)
            except (ValueError, TypeError):
                print(f"Warning: Could not convert column '{column}' to {dtype}")
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
    
    numeric_cols = df_copy[columns].select_dtypes(include=[np.number]).columns
    
    if strategy == 'drop':
        return df_copy.dropna(subset=columns)
    
    for col in columns:
        if col in numeric_cols:
            if strategy == 'mean':
                fill_value = df_copy[col].mean()
            elif strategy == 'median':
                fill_value = df_copy[col].median()
            elif strategy == 'mode':
                fill_value = df_copy[col].mode()[0] if not df_copy[col].mode().empty else 0
            else:
                fill_value = 0
        else:
            fill_value = df_copy[col].mode()[0] if not df_copy[col].mode().empty else ''
        
        df_copy[col] = df_copy[col].fillna(fill_value)
    
    return df_copy

def normalize_column(df: pd.DataFrame, 
                    column: str,
                    method: str = 'minmax') -> pd.DataFrame:
    """
    Normalize a numeric column.
    
    Args:
        df: Input DataFrame
        column: Column to normalize
        method: Normalization method ('minmax' or 'zscore')
    
    Returns:
        DataFrame with normalized column
    """
    df_copy = df.copy()
    
    if column not in df_copy.columns:
        return df_copy
    
    if method == 'minmax':
        col_min = df_copy[column].min()
        col_max = df_copy[column].max()
        if col_max != col_min:
            df_copy[column] = (df_copy[column] - col_min) / (col_max - col_min)
    
    elif method == 'zscore':
        col_mean = df_copy[column].mean()
        col_std = df_copy[column].std()
        if col_std > 0:
            df_copy[column] = (df_copy[column] - col_mean) / col_std
    
    return df_copy

def clean_dataframe(df: pd.DataFrame,
                   deduplicate: bool = True,
                   type_conversions: dict = None,
                   missing_strategy: str = 'mean',
                   normalize_columns: List[str] = None) -> pd.DataFrame:
    """
    Comprehensive data cleaning pipeline.
    
    Args:
        df: Input DataFrame
        deduplicate: Whether to remove duplicates
        type_conversions: Dictionary for column type conversions
        missing_strategy: Strategy for handling missing values
        normalize_columns: Columns to normalize
    
    Returns:
        Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    if deduplicate:
        cleaned_df = remove_duplicates(cleaned_df)
    
    if type_conversions:
        cleaned_df = convert_column_types(cleaned_df, type_conversions)
    
    cleaned_df = handle_missing_values(cleaned_df, strategy=missing_strategy)
    
    if normalize_columns:
        for col in normalize_columns:
            if col in cleaned_df.columns:
                cleaned_df = normalize_column(cleaned_df, col)
    
    return cleaned_df