import pandas as pd
import numpy as np

def remove_missing_rows(df, threshold=0.5):
    """
    Remove rows with missing values exceeding the threshold percentage.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        threshold (float): Maximum allowed missing value percentage per row (0-1)
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    missing_percentage = df.isnull().mean(axis=1)
    return df[missing_percentage <= threshold].reset_index(drop=True)

def replace_outliers_iqr(df, columns=None, multiplier=1.5):
    """
    Replace outliers with column boundaries using IQR method.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        columns (list): List of column names to process, None for all numeric columns
        multiplier (float): IQR multiplier for outlier detection
    
    Returns:
        pd.DataFrame: DataFrame with outliers replaced
    """
    df_clean = df.copy()
    
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    for col in columns:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - multiplier * IQR
            upper_bound = Q3 + multiplier * IQR
            
            df_clean[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
    
    return df_clean

def standardize_columns(df, columns=None):
    """
    Standardize specified columns using z-score normalization.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        columns (list): List of column names to standardize
    
    Returns:
        pd.DataFrame: DataFrame with standardized columns
    """
    df_standardized = df.copy()
    
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    for col in columns:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            mean_val = df[col].mean()
            std_val = df[col].std()
            
            if std_val > 0:
                df_standardized[col] = (df[col] - mean_val) / std_val
    
    return df_standardized

def clean_dataset(df, missing_threshold=0.3, outlier_multiplier=1.5, standardize=True):
    """
    Complete data cleaning pipeline.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        missing_threshold (float): Threshold for removing rows with missing values
        outlier_multiplier (float): Multiplier for IQR outlier detection
        standardize (bool): Whether to standardize numeric columns
    
    Returns:
        pd.DataFrame: Cleaned and processed DataFrame
    """
    print(f"Original shape: {df.shape}")
    
    # Step 1: Handle missing values
    df_clean = remove_missing_rows(df, threshold=missing_threshold)
    print(f"After missing value removal: {df_clean.shape}")
    
    # Step 2: Handle outliers
    df_clean = replace_outliers_iqr(df_clean, multiplier=outlier_multiplier)
    print("Outliers replaced using IQR method")
    
    # Step 3: Standardize if requested
    if standardize:
        df_clean = standardize_columns(df_clean)
        print("Numeric columns standardized")
    
    return df_clean