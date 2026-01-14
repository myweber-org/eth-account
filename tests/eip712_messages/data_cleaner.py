import pandas as pd
import numpy as np
from typing import Optional

def remove_duplicates(df: pd.DataFrame, subset: Optional[list] = None) -> pd.DataFrame:
    """
    Remove duplicate rows from DataFrame.
    """
    return df.drop_duplicates(subset=subset, keep='first')

def fill_missing_values(df: pd.DataFrame, strategy: str = 'mean', columns: Optional[list] = None) -> pd.DataFrame:
    """
    Fill missing values using specified strategy.
    """
    df_filled = df.copy()
    cols_to_fill = columns if columns else df.columns
    
    for col in cols_to_fill:
        if df[col].dtype in ['int64', 'float64']:
            if strategy == 'mean':
                df_filled[col] = df[col].fillna(df[col].mean())
            elif strategy == 'median':
                df_filled[col] = df[col].fillna(df[col].median())
            elif strategy == 'mode':
                df_filled[col] = df[col].fillna(df[col].mode()[0])
            elif strategy == 'zero':
                df_filled[col] = df[col].fillna(0)
        else:
            df_filled[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown')
    
    return df_filled

def normalize_column(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Normalize a numeric column to range [0, 1].
    """
    df_normalized = df.copy()
    if df[column].dtype in ['int64', 'float64']:
        col_min = df[column].min()
        col_max = df[column].max()
        if col_max != col_min:
            df_normalized[column] = (df[column] - col_min) / (col_max - col_min)
    return df_normalized

def clean_dataframe(df: pd.DataFrame, 
                    remove_dups: bool = True,
                    fill_na: bool = True,
                    normalize_cols: Optional[list] = None) -> pd.DataFrame:
    """
    Main cleaning pipeline for DataFrame.
    """
    cleaned_df = df.copy()
    
    if remove_dups:
        cleaned_df = remove_duplicates(cleaned_df)
    
    if fill_na:
        cleaned_df = fill_missing_values(cleaned_df)
    
    if normalize_cols:
        for col in normalize_cols:
            if col in cleaned_df.columns:
                cleaned_df = normalize_column(cleaned_df, col)
    
    return cleaned_df

def load_and_clean_csv(filepath: str, **kwargs) -> pd.DataFrame:
    """
    Load CSV file and apply cleaning pipeline.
    """
    try:
        df = pd.read_csv(filepath)
        return clean_dataframe(df, **kwargs)
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error loading file: {e}")
        return pd.DataFrame()