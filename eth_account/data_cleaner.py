
import pandas as pd
import numpy as np

def clean_missing_values(df, strategy='mean', columns=None):
    """
    Handle missing values in a DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        strategy (str): Strategy for imputation ('mean', 'median', 'mode', 'drop')
        columns (list): Specific columns to clean, if None clean all columns
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    df_clean = df.copy()
    
    if columns is None:
        columns = df_clean.columns
    
    for col in columns:
        if df_clean[col].isnull().any():
            if strategy == 'mean':
                df_clean[col].fillna(df_clean[col].mean(), inplace=True)
            elif strategy == 'median':
                df_clean[col].fillna(df_clean[col].median(), inplace=True)
            elif strategy == 'mode':
                df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)
            elif strategy == 'drop':
                df_clean = df_clean.dropna(subset=[col])
    
    return df_clean

def remove_outliers_iqr(df, columns=None, threshold=1.5):
    """
    Remove outliers using the Interquartile Range method.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        columns (list): Columns to check for outliers
        threshold (float): IQR multiplier threshold
    
    Returns:
        pd.DataFrame: DataFrame with outliers removed
    """
    df_clean = df.copy()
    
    if columns is None:
        columns = df_clean.select_dtypes(include=[np.number]).columns
    
    for col in columns:
        if df_clean[col].dtype in [np.float64, np.int64]:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            mask = (df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)
            df_clean = df_clean[mask]
    
    return df_clean.reset_index(drop=True)

def standardize_columns(df, columns=None):
    """
    Standardize numerical columns to have zero mean and unit variance.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        columns (list): Columns to standardize
    
    Returns:
        pd.DataFrame: DataFrame with standardized columns
    """
    df_standardized = df.copy()
    
    if columns is None:
        columns = df_standardized.select_dtypes(include=[np.number]).columns
    
    for col in columns:
        if df_standardized[col].dtype in [np.float64, np.int64]:
            mean = df_standardized[col].mean()
            std = df_standardized[col].std()
            if std > 0:
                df_standardized[col] = (df_standardized[col] - mean) / std
    
    return df_standardized

def validate_dataframe(df, required_columns=None, min_rows=1):
    """
    Validate DataFrame structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        required_columns (list): List of required column names
        min_rows (int): Minimum number of rows required
    
    Returns:
        tuple: (is_valid, error_message)
    """
    if not isinstance(df, pd.DataFrame):
        return False, "Input is not a pandas DataFrame"
    
    if len(df) < min_rows:
        return False, f"DataFrame has fewer than {min_rows} rows"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    return True, "DataFrame is valid"