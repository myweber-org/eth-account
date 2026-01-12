import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a specified column in a DataFrame using the IQR method.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    column (str): The column name to process.
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed.
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

def clean_dataset(df, numeric_columns):
    """
    Clean a dataset by removing outliers from multiple numeric columns.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    numeric_columns (list): List of column names to clean.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    if not isinstance(numeric_columns, list):
        raise TypeError("numeric_columns must be a list")
    
    cleaned_df = df.copy()
    
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
    
    return cleaned_df

def get_cleaning_stats(original_df, cleaned_df):
    """
    Calculate statistics about the cleaning process.
    
    Parameters:
    original_df (pd.DataFrame): Original DataFrame before cleaning.
    cleaned_df (pd.DataFrame): Cleaned DataFrame after outlier removal.
    
    Returns:
    dict: Dictionary containing cleaning statistics.
    """
    stats = {
        'original_rows': len(original_df),
        'cleaned_rows': len(cleaned_df),
        'rows_removed': len(original_df) - len(cleaned_df),
        'removal_percentage': ((len(original_df) - len(cleaned_df)) / len(original_df)) * 100
    }
    
    return stats