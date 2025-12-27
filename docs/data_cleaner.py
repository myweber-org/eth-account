import numpy as np
import pandas as pd

def remove_outliers_iqr(data, column, factor=1.5):
    """
    Remove outliers using the Interquartile Range method.
    
    Args:
        data: pandas DataFrame
        column: column name to process
        factor: IQR multiplier (default 1.5)
    
    Returns:
        DataFrame with outliers removed
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    
    lower_bound = q1 - factor * iqr
    upper_bound = q3 + factor * iqr
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data

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

def clean_missing_values(data, strategy='mean', columns=None):
    """
    Handle missing values using specified strategy.
    
    Args:
        data: pandas DataFrame
        strategy: 'mean', 'median', 'mode', or 'drop'
        columns: list of columns to process (None for all numeric columns)
    
    Returns:
        DataFrame with missing values handled
    """
    if columns is None:
        columns = data.select_dtypes(include=[np.number]).columns
    
    cleaned_data = data.copy()
    
    for col in columns:
        if col not in cleaned_data.columns:
            continue
            
        if strategy == 'drop':
            cleaned_data = cleaned_data.dropna(subset=[col])
        elif strategy == 'mean':
            cleaned_data[col] = cleaned_data[col].fillna(cleaned_data[col].mean())
        elif strategy == 'median':
            cleaned_data[col] = cleaned_data[col].fillna(cleaned_data[col].median())
        elif strategy == 'mode':
            cleaned_data[col] = cleaned_data[col].fillna(cleaned_data[col].mode()[0])
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    return cleaned_data

def process_dataset(df, numeric_columns, outlier_factor=1.5, normalize_cols=None, standardize_cols=None):
    """
    Complete data cleaning pipeline.
    
    Args:
        df: input DataFrame
        numeric_columns: list of numeric columns to process
        outlier_factor: IQR factor for outlier removal
        normalize_cols: columns to min-max normalize
        standardize_cols: columns to z-score standardize
    
    Returns:
        Cleaned and processed DataFrame
    """
    processed_df = df.copy()
    
    for col in numeric_columns:
        if col in processed_df.columns:
            processed_df = remove_outliers_iqr(processed_df, col, outlier_factor)
    
    processed_df = clean_missing_values(processed_df, strategy='median', columns=numeric_columns)
    
    if normalize_cols:
        for col in normalize_cols:
            if col in processed_df.columns:
                processed_df[f'{col}_normalized'] = normalize_minmax(processed_df, col)
    
    if standardize_cols:
        for col in standardize_cols:
            if col in processed_df.columns:
                processed_df[f'{col}_standardized'] = standardize_zscore(processed_df, col)
    
    return processed_df