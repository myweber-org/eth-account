import re
import string

def remove_punctuation(text):
    """Remove all punctuation from the input string."""
    return text.translate(str.maketrans('', '', string.punctuation))

def normalize_whitespace(text):
    """Replace multiple whitespace characters with a single space."""
    return re.sub(r'\s+', ' ', text).strip()

def clean_text(text, remove_punct=True, normalize_ws=True):
    """Clean text by optionally removing punctuation and normalizing whitespace."""
    if remove_punct:
        text = remove_punctuation(text)
    if normalize_ws:
        text = normalize_whitespace(text)
    return text.lower()

def tokenize_text(text, delimiter=' '):
    """Split text into tokens based on the specified delimiter."""
    return text.split(delimiter)

def process_text_list(text_list, **kwargs):
    """Apply cleaning functions to a list of text strings."""
    return [clean_text(text, **kwargs) for text in text_list]import pandas as pd
import numpy as np

def remove_missing_rows(df, columns=None):
    """
    Remove rows with missing values from DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        columns (list, optional): Specific columns to check for missing values.
                                 If None, checks all columns.
    
    Returns:
        pd.DataFrame: DataFrame with rows containing missing values removed
    """
    if columns is None:
        return df.dropna()
    else:
        return df.dropna(subset=columns)

def fill_missing_with_mean(df, columns=None):
    """
    Fill missing values with column mean.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        columns (list, optional): Specific columns to fill.
                                 If None, fills all numeric columns.
    
    Returns:
        pd.DataFrame: DataFrame with missing values filled
    """
    df_copy = df.copy()
    
    if columns is None:
        numeric_cols = df_copy.select_dtypes(include=[np.number]).columns
        columns = list(numeric_cols)
    
    for col in columns:
        if col in df_copy.columns and pd.api.types.is_numeric_dtype(df_copy[col]):
            mean_val = df_copy[col].mean()
            df_copy[col] = df_copy[col].fillna(mean_val)
    
    return df_copy

def remove_outliers_iqr(df, column, multiplier=1.5):
    """
    Remove outliers using IQR method.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        column (str): Column name to check for outliers
        multiplier (float): IQR multiplier for outlier detection
    
    Returns:
        pd.DataFrame: DataFrame with outliers removed
    """
    if column not in df.columns:
        return df
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def standardize_columns(df, columns=None):
    """
    Standardize numeric columns to have zero mean and unit variance.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        columns (list, optional): Specific columns to standardize.
                                 If None, standardizes all numeric columns.
    
    Returns:
        pd.DataFrame: DataFrame with standardized columns
    """
    df_copy = df.copy()
    
    if columns is None:
        numeric_cols = df_copy.select_dtypes(include=[np.number]).columns
        columns = list(numeric_cols)
    
    for col in columns:
        if col in df_copy.columns and pd.api.types.is_numeric_dtype(df_copy[col]):
            mean_val = df_copy[col].mean()
            std_val = df_copy[col].std()
            
            if std_val > 0:
                df_copy[col] = (df_copy[col] - mean_val) / std_val
    
    return df_copy

def clean_dataset(df, missing_strategy='remove', outlier_columns=None):
    """
    Comprehensive data cleaning function.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        missing_strategy (str): Strategy for handling missing values.
                               Options: 'remove', 'mean'
        outlier_columns (list, optional): Columns to check for outliers.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    # Handle missing values
    if missing_strategy == 'remove':
        cleaned_df = remove_missing_rows(cleaned_df)
    elif missing_strategy == 'mean':
        cleaned_df = fill_missing_with_mean(cleaned_df)
    
    # Remove outliers if specified
    if outlier_columns:
        for col in outlier_columns:
            if col in cleaned_df.columns:
                cleaned_df = remove_outliers_iqr(cleaned_df, col)
    
    # Standardize numeric columns
    cleaned_df = standardize_columns(cleaned_df)
    
    return cleaned_df