
import numpy as np
import pandas as pd

def remove_outliers_iqr(data, column, multiplier=1.5):
    """
    Remove outliers using the Interquartile Range method.
    
    Args:
        data: pandas DataFrame
        column: column name to process
        multiplier: IQR multiplier (default 1.5)
    
    Returns:
        Filtered DataFrame without outliers
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data.copy()

def normalize_minmax(data, column):
    """
    Normalize data to [0, 1] range using min-max scaling.
    
    Args:
        data: pandas DataFrame
        column: column name to normalize
    
    Returns:
        Series with normalized values
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    min_val = data[column].min()
    max_val = data[column].max()
    
    if max_val == min_val:
        return pd.Series([0.5] * len(data), index=data.index)
    
    normalized = (data[column] - min_val) / (max_val - min_val)
    return normalized

def standardize_zscore(data, column):
    """
    Standardize data using z-score normalization.
    
    Args:
        data: pandas DataFrame
        column: column name to standardize
    
    Returns:
        Series with standardized values
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    mean_val = data[column].mean()
    std_val = data[column].std()
    
    if std_val == 0:
        return pd.Series([0] * len(data), index=data.index)
    
    standardized = (data[column] - mean_val) / std_val
    return standardized

def clean_missing_values(data, strategy='mean'):
    """
    Handle missing values in numeric columns.
    
    Args:
        data: pandas DataFrame
        strategy: imputation strategy ('mean', 'median', 'mode', or 'drop')
    
    Returns:
        DataFrame with handled missing values
    """
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    
    if strategy == 'drop':
        cleaned_data = data.dropna(subset=numeric_cols)
    elif strategy == 'mean':
        cleaned_data = data.copy()
        for col in numeric_cols:
            cleaned_data[col].fillna(data[col].mean(), inplace=True)
    elif strategy == 'median':
        cleaned_data = data.copy()
        for col in numeric_cols:
            cleaned_data[col].fillna(data[col].median(), inplace=True)
    elif strategy == 'mode':
        cleaned_data = data.copy()
        for col in numeric_cols:
            cleaned_data[col].fillna(data[col].mode()[0], inplace=True)
    else:
        raise ValueError("Strategy must be 'mean', 'median', 'mode', or 'drop'")
    
    return cleaned_data

def process_dataframe(df, numeric_columns=None, outlier_multiplier=1.5, 
                     normalization_method='standardize', missing_strategy='mean'):
    """
    Complete data cleaning pipeline for numeric columns.
    
    Args:
        df: Input DataFrame
        numeric_columns: list of numeric columns to process (default: all numeric)
        outlier_multiplier: multiplier for IQR outlier detection
        normalization_method: 'standardize', 'normalize', or None
        missing_strategy: strategy for handling missing values
    
    Returns:
        Cleaned and processed DataFrame
    """
    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    processed_df = df.copy()
    
    # Handle missing values
    processed_df = clean_missing_values(processed_df, strategy=missing_strategy)
    
    # Remove outliers for each numeric column
    for col in numeric_columns:
        if col in processed_df.columns:
            processed_df = remove_outliers_iqr(processed_df, col, outlier_multiplier)
    
    # Apply normalization
    for col in numeric_columns:
        if col in processed_df.columns:
            if normalization_method == 'standardize':
                processed_df[f'{col}_standardized'] = standardize_zscore(processed_df, col)
            elif normalization_method == 'normalize':
                processed_df[f'{col}_normalized'] = normalize_minmax(processed_df, col)
    
    return processed_df