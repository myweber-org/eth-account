
import pandas as pd
import numpy as np

def clean_missing_values(df, strategy='mean', columns=None):
    """
    Handle missing values in a DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        strategy (str): Strategy for imputation ('mean', 'median', 'mode', 'drop')
        columns (list): List of columns to process, if None processes all numeric columns
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    df_clean = df.copy()
    
    if columns is None:
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        columns = list(numeric_cols)
    
    if strategy == 'drop':
        df_clean = df_clean.dropna(subset=columns)
    else:
        for col in columns:
            if col in df_clean.columns:
                if strategy == 'mean':
                    fill_value = df_clean[col].mean()
                elif strategy == 'median':
                    fill_value = df_clean[col].median()
                elif strategy == 'mode':
                    fill_value = df_clean[col].mode()[0]
                else:
                    raise ValueError(f"Unknown strategy: {strategy}")
                
                df_clean[col] = df_clean[col].fillna(fill_value)
    
    return df_clean

def remove_outliers_iqr(df, columns=None, multiplier=1.5):
    """
    Remove outliers using the Interquartile Range method.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        columns (list): List of columns to process, if None processes all numeric columns
        multiplier (float): IQR multiplier for outlier detection
    
    Returns:
        pd.DataFrame: DataFrame with outliers removed
    """
    df_clean = df.copy()
    
    if columns is None:
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        columns = list(numeric_cols)
    
    mask = pd.Series([True] * len(df_clean))
    
    for col in columns:
        if col in df_clean.columns:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - multiplier * IQR
            upper_bound = Q3 + multiplier * IQR
            
            col_mask = (df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)
            mask = mask & col_mask
    
    return df_clean[mask]

def standardize_columns(df, columns=None):
    """
    Standardize numeric columns to have zero mean and unit variance.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        columns (list): List of columns to standardize
    
    Returns:
        pd.DataFrame: DataFrame with standardized columns
    """
    df_standardized = df.copy()
    
    if columns is None:
        numeric_cols = df_standardized.select_dtypes(include=[np.number]).columns
        columns = list(numeric_cols)
    
    for col in columns:
        if col in df_standardized.columns:
            mean_val = df_standardized[col].mean()
            std_val = df_standardized[col].std()
            
            if std_val > 0:
                df_standardized[col] = (df_standardized[col] - mean_val) / std_val
    
    return df_standardized

def clean_dataset(df, missing_strategy='mean', remove_outliers=True, standardize=True):
    """
    Comprehensive data cleaning pipeline.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        missing_strategy (str): Strategy for handling missing values
        remove_outliers (bool): Whether to remove outliers
        standardize (bool): Whether to standardize numeric columns
    
    Returns:
        pd.DataFrame: Cleaned and processed DataFrame
    """
    cleaned_df = df.copy()
    
    cleaned_df = clean_missing_values(cleaned_df, strategy=missing_strategy)
    
    if remove_outliers:
        cleaned_df = remove_outliers_iqr(cleaned_df)
    
    if standardize:
        cleaned_df = standardize_columns(cleaned_df)
    
    return cleaned_df