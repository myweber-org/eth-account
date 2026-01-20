import pandas as pd
import numpy as np

def remove_outliers_iqr(df, columns=None):
    """
    Remove outliers from specified columns using the Interquartile Range method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    columns (list): List of column names to process. If None, process all numeric columns.
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    df_clean = df.copy()
    
    for col in columns:
        if col not in df.columns:
            continue
            
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        mask = (df[col] >= lower_bound) & (df[col] <= upper_bound)
        df_clean = df_clean[mask]
    
    return df_clean.reset_index(drop=True)

def normalize_data(df, columns=None, method='minmax'):
    """
    Normalize specified columns using selected method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    columns (list): List of column names to normalize
    method (str): Normalization method ('minmax' or 'zscore')
    
    Returns:
    pd.DataFrame: DataFrame with normalized columns
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    df_norm = df.copy()
    
    for col in columns:
        if col not in df.columns:
            continue
            
        if method == 'minmax':
            min_val = df[col].min()
            max_val = df[col].max()
            if max_val != min_val:
                df_norm[col] = (df[col] - min_val) / (max_val - min_val)
            else:
                df_norm[col] = 0
        
        elif method == 'zscore':
            mean_val = df[col].mean()
            std_val = df[col].std()
            if std_val != 0:
                df_norm[col] = (df[col] - mean_val) / std_val
            else:
                df_norm[col] = 0
    
    return df_norm

def handle_missing_values(df, strategy='mean', columns=None):
    """
    Handle missing values in specified columns.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    strategy (str): Strategy for handling missing values ('mean', 'median', 'mode', 'drop')
    columns (list): List of column names to process
    
    Returns:
    pd.DataFrame: DataFrame with handled missing values
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    df_processed = df.copy()
    
    for col in columns:
        if col not in df.columns:
            continue
            
        if strategy == 'mean':
            fill_value = df[col].mean()
        elif strategy == 'median':
            fill_value = df[col].median()
        elif strategy == 'mode':
            fill_value = df[col].mode()[0] if not df[col].mode().empty else 0
        elif strategy == 'drop':
            df_processed = df_processed.dropna(subset=[col])
            continue
        
        df_processed[col] = df_processed[col].fillna(fill_value)
    
    return df_processed.reset_index(drop=True)