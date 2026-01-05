import pandas as pd
import numpy as np
from typing import List, Optional

def remove_duplicates(df: pd.DataFrame, subset: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Remove duplicate rows from DataFrame.
    
    Args:
        df: Input DataFrame
        subset: Columns to consider for identifying duplicates
    
    Returns:
        DataFrame with duplicates removed
    """
    return df.drop_duplicates(subset=subset, keep='first')

def normalize_column(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Normalize a column to have values between 0 and 1.
    
    Args:
        df: Input DataFrame
        column: Column name to normalize
    
    Returns:
        DataFrame with normalized column
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    col_min = df[column].min()
    col_max = df[column].max()
    
    if col_max == col_min:
        df[column] = 0.5
    else:
        df[column] = (df[column] - col_min) / (col_max - col_min)
    
    return df

def clean_missing_values(df: pd.DataFrame, strategy: str = 'mean') -> pd.DataFrame:
    """
    Handle missing values in DataFrame.
    
    Args:
        df: Input DataFrame
        strategy: Method for handling missing values ('mean', 'median', 'mode', 'drop')
    
    Returns:
        DataFrame with handled missing values
    """
    df_clean = df.copy()
    
    if strategy == 'drop':
        return df_clean.dropna()
    
    for col in df_clean.columns:
        if df_clean[col].dtype in [np.float64, np.int64]:
            if strategy == 'mean':
                fill_value = df_clean[col].mean()
            elif strategy == 'median':
                fill_value = df_clean[col].median()
            elif strategy == 'mode':
                fill_value = df_clean[col].mode()[0] if not df_clean[col].mode().empty else 0
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
            
            df_clean[col].fillna(fill_value, inplace=True)
    
    return df_clean

def validate_dataframe(df: pd.DataFrame) -> bool:
    """
    Perform basic validation on DataFrame.
    
    Args:
        df: DataFrame to validate
    
    Returns:
        True if DataFrame passes validation
    """
    if df.empty:
        raise ValueError("DataFrame is empty")
    
    if df.isnull().all().any():
        raise ValueError("Some columns contain only null values")
    
    return True