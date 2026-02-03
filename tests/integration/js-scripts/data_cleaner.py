
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

def clean_missing_values(df, strategy='mean', columns=None):
    """
    Handle missing values in specified columns.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    strategy (str): Imputation strategy ('mean', 'median', 'mode', or 'drop').
    columns (list): List of column names to process. If None, process all columns.
    
    Returns:
    pd.DataFrame: DataFrame with missing values handled.
    """
    if columns is None:
        columns = df.columns
    
    df_clean = df.copy()
    
    for col in columns:
        if col not in df.columns:
            continue
            
        if strategy == 'drop':
            df_clean = df_clean.dropna(subset=[col])
        elif strategy == 'mean':
            df_clean[col].fillna(df_clean[col].mean(), inplace=True)
        elif strategy == 'median':
            df_clean[col].fillna(df_clean[col].median(), inplace=True)
        elif strategy == 'mode':
            df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    return df_clean

def normalize_column(df, column):
    """
    Normalize a column using min-max scaling.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    column (str): The column name to normalize.
    
    Returns:
    pd.DataFrame: DataFrame with normalized column.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    df_normalized = df.copy()
    min_val = df_normalized[column].min()
    max_val = df_normalized[column].max()
    
    if max_val != min_val:
        df_normalized[column] = (df_normalized[column] - min_val) / (max_val - min_val)
    
    return df_normalized