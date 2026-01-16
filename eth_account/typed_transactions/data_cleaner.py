import pandas as pd
import numpy as np

def remove_duplicates(df, subset=None):
    """
    Remove duplicate rows from a DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        subset (list, optional): Columns to consider for duplicates.
    
    Returns:
        pd.DataFrame: DataFrame with duplicates removed.
    """
    return df.drop_duplicates(subset=subset, keep='first')

def fill_missing_values(df, strategy='mean', columns=None):
    """
    Fill missing values in specified columns.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        strategy (str): 'mean', 'median', 'mode', or 'constant'.
        columns (list): Columns to fill. If None, fill all numeric columns.
    
    Returns:
        pd.DataFrame: DataFrame with missing values filled.
    """
    df_filled = df.copy()
    
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    for col in columns:
        if strategy == 'mean':
            df_filled[col].fillna(df[col].mean(), inplace=True)
        elif strategy == 'median':
            df_filled[col].fillna(df[col].median(), inplace=True)
        elif strategy == 'mode':
            df_filled[col].fillna(df[col].mode()[0], inplace=True)
        elif strategy == 'constant':
            df_filled[col].fillna(0, inplace=True)
    
    return df_filled

def normalize_column(df, column, method='minmax'):
    """
    Normalize a column using specified method.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        column (str): Column name to normalize.
        method (str): 'minmax' or 'zscore'.
    
    Returns:
        pd.DataFrame: DataFrame with normalized column.
    """
    df_normalized = df.copy()
    
    if method == 'minmax':
        min_val = df[column].min()
        max_val = df[column].max()
        if max_val != min_val:
            df_normalized[column] = (df[column] - min_val) / (max_val - min_val)
    
    elif method == 'zscore':
        mean_val = df[column].mean()
        std_val = df[column].std()
        if std_val != 0:
            df_normalized[column] = (df[column] - mean_val) / std_val
    
    return df_normalized

def filter_outliers(df, column, method='iqr', threshold=1.5):
    """
    Filter outliers from a DataFrame column.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        column (str): Column name to filter.
        method (str): 'iqr' or 'zscore'.
        threshold (float): Threshold for outlier detection.
    
    Returns:
        pd.DataFrame: DataFrame with outliers removed.
    """
    if method == 'iqr':
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
    elif method == 'zscore':
        z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
        return df[z_scores <= threshold]
    
    return df

def clean_dataframe(df, operations):
    """
    Apply multiple cleaning operations sequentially.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        operations (list): List of cleaning operations to apply.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    for operation in operations:
        if operation['type'] == 'remove_duplicates':
            cleaned_df = remove_duplicates(cleaned_df, **operation.get('params', {}))
        elif operation['type'] == 'fill_missing':
            cleaned_df = fill_missing_values(cleaned_df, **operation.get('params', {}))
        elif operation['type'] == 'normalize':
            cleaned_df = normalize_column(cleaned_df, **operation.get('params', {}))
        elif operation['type'] == 'filter_outliers':
            cleaned_df = filter_outliers(cleaned_df, **operation.get('params', {}))
    
    return cleaned_df