
import re
import pandas as pd
from typing import Union, List, Optional

def remove_special_chars(text: str, keep_chars: str = '') -> str:
    """
    Remove all special characters from a string, optionally keeping some.
    """
    if not isinstance(text, str):
        return text
    pattern = f'[^A-Za-z0-9\\s{re.escape(keep_chars)}]'
    return re.sub(pattern, '', text)

def normalize_whitespace(text: str) -> str:
    """
    Replace multiple whitespace characters with a single space.
    """
    if not isinstance(text, str):
        return text
    return ' '.join(text.split())

def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean column names of a DataFrame: lowercase, replace spaces with underscores.
    """
    df.columns = [col.lower().replace(' ', '_') for col in df.columns]
    return df

def drop_missing_rows(df: pd.DataFrame, columns: Optional[List[str]] = None, threshold: float = 0.5) -> pd.DataFrame:
    """
    Drop rows with missing values above a threshold.
    """
    if columns is None:
        columns = df.columns
    missing_threshold = len(columns) * threshold
    return df[df[columns].isnull().sum(axis=1) <= missing_threshold]

def convert_to_numeric(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    Convert specified columns to numeric, coercing errors to NaN.
    """
    df_copy = df.copy()
    for col in columns:
        df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')
    return df_copy

def remove_outliers_iqr(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Remove outliers from a column using the IQR method.
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]