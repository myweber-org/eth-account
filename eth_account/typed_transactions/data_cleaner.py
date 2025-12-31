import numpy as np
import pandas as pd

def remove_outliers_iqr(df, columns, factor=1.5):
    """
    Remove outliers using IQR method.
    
    Parameters:
    df (pd.DataFrame): Input dataframe
    columns (list): List of column names to process
    factor (float): IQR multiplier for outlier detection
    
    Returns:
    pd.DataFrame: Dataframe with outliers removed
    """
    df_clean = df.copy()
    
    for col in columns:
        if col not in df_clean.columns:
            continue
            
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR
        
        mask = (df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)
        df_clean = df_clean[mask]
    
    return df_clean.reset_index(drop=True)

def normalize_minmax(df, columns):
    """
    Normalize columns using min-max scaling.
    
    Parameters:
    df (pd.DataFrame): Input dataframe
    columns (list): List of column names to normalize
    
    Returns:
    pd.DataFrame: Dataframe with normalized columns
    """
    df_norm = df.copy()
    
    for col in columns:
        if col not in df_norm.columns:
            continue
            
        min_val = df_norm[col].min()
        max_val = df_norm[col].max()
        
        if max_val != min_val:
            df_norm[col] = (df_norm[col] - min_val) / (max_val - min_val)
        else:
            df_norm[col] = 0
    
    return df_norm

def handle_missing_values(df, strategy='mean', columns=None):
    """
    Handle missing values in dataframe.
    
    Parameters:
    df (pd.DataFrame): Input dataframe
    strategy (str): Strategy for imputation ('mean', 'median', 'mode', 'drop')
    columns (list): List of columns to process, None for all numeric columns
    
    Returns:
    pd.DataFrame: Dataframe with handled missing values
    """
    df_processed = df.copy()
    
    if columns is None:
        columns = df_processed.select_dtypes(include=[np.number]).columns.tolist()
    
    if strategy == 'drop':
        df_processed = df_processed.dropna(subset=columns)
    else:
        for col in columns:
            if col not in df_processed.columns:
                continue
                
            if df_processed[col].isnull().any():
                if strategy == 'mean':
                    fill_value = df_processed[col].mean()
                elif strategy == 'median':
                    fill_value = df_processed[col].median()
                elif strategy == 'mode':
                    fill_value = df_processed[col].mode()[0]
                else:
                    fill_value = 0
                
                df_processed[col] = df_processed[col].fillna(fill_value)
    
    return df_processed.reset_index(drop=True)

def clean_dataset(df, numeric_columns, outlier_factor=1.5, normalize=True, missing_strategy='mean'):
    """
    Complete data cleaning pipeline.
    
    Parameters:
    df (pd.DataFrame): Input dataframe
    numeric_columns (list): List of numeric column names to process
    outlier_factor (float): IQR multiplier for outlier detection
    normalize (bool): Whether to apply min-max normalization
    missing_strategy (str): Strategy for handling missing values
    
    Returns:
    pd.DataFrame: Cleaned dataframe
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame")
    
    if not numeric_columns:
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    df_clean = handle_missing_values(df, strategy=missing_strategy, columns=numeric_columns)
    df_clean = remove_outliers_iqr(df_clean, numeric_columns, factor=outlier_factor)
    
    if normalize:
        df_clean = normalize_minmax(df_clean, numeric_columns)
    
    return df_clean