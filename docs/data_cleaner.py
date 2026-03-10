import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column, factor=1.5):
    """
    Remove outliers from a DataFrame column using the IQR method.
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - factor * IQR
    upper_bound = Q3 + factor * IQR
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data.copy()

def normalize_column(data, column, method='zscore'):
    """
    Normalize a column using specified method.
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    normalized_data = data.copy()
    
    if method == 'zscore':
        normalized_data[column] = stats.zscore(data[column])
    elif method == 'minmax':
        col_min = data[column].min()
        col_max = data[column].max()
        normalized_data[column] = (data[column] - col_min) / (col_max - col_min)
    elif method == 'robust':
        median = data[column].median()
        iqr = data[column].quantile(0.75) - data[column].quantile(0.25)
        normalized_data[column] = (data[column] - median) / iqr
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return normalized_data

def clean_dataset(df, numeric_columns=None, outlier_factor=1.5, normalize_method='zscore'):
    """
    Apply cleaning pipeline to dataset.
    """
    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_df = df.copy()
    
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, col, outlier_factor)
            cleaned_df = normalize_column(cleaned_df, col, normalize_method)
    
    return cleaned_df

def validate_dataframe(df, required_columns=None, allow_nan=False):
    """
    Validate DataFrame structure and content.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
    
    if not allow_nan and df.isnull().any().any():
        raise ValueError("DataFrame contains NaN values")
    
    return True