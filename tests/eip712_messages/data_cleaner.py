import numpy as np
import pandas as pd

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a pandas Series using the IQR method.
    Returns a cleaned Series with outliers set to NaN.
    """
    if not isinstance(data, pd.Series):
        raise TypeError("Input data must be a pandas Series")
    
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    cleaned_data = data.copy()
    cleaned_data[(cleaned_data < lower_bound) | (cleaned_data > upper_bound)] = np.nan
    return cleaned_data

def normalize_minmax(data):
    """
    Normalize data using min-max scaling to range [0, 1].
    Handles NaN values by ignoring them in calculation.
    """
    if not isinstance(data, pd.Series):
        raise TypeError("Input data must be a pandas Series")
    
    min_val = data.min(skipna=True)
    max_val = data.max(skipna=True)
    
    if min_val == max_val:
        return pd.Series([0.5] * len(data), index=data.index)
    
    normalized = (data - min_val) / (max_val - min_val)
    return normalized

def clean_dataframe(df, numeric_columns=None):
    """
    Clean a DataFrame by removing outliers and normalizing numeric columns.
    If numeric_columns is None, automatically select all numeric columns.
    Returns a new cleaned DataFrame.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    
    cleaned_df = df.copy()
    
    if numeric_columns is None:
        numeric_columns = cleaned_df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_columns:
        if col in cleaned_df.columns:
            # Remove outliers
            cleaned_df[col] = remove_outliers_iqr(cleaned_df[col], col)
            # Normalize remaining values
            cleaned_df[col] = normalize_minmax(cleaned_df[col])
    
    return cleaned_df

def calculate_statistics(df):
    """
    Calculate basic statistics for numeric columns in DataFrame.
    Returns a summary DataFrame with count, mean, std, min, and max.
    """
    numeric_df = df.select_dtypes(include=[np.number])
    stats = pd.DataFrame({
        'count': numeric_df.count(),
        'mean': numeric_df.mean(),
        'std': numeric_df.std(),
        'min': numeric_df.min(),
        'max': numeric_df.max()
    })
    return stats.T