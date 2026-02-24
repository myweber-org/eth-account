
import numpy as np
import pandas as pd

def remove_outliers_iqr(data, column, multiplier=1.5):
    """
    Remove outliers using the Interquartile Range method.
    
    Args:
        data: pandas DataFrame
        column: column name to process
        multiplier: IQR multiplier for outlier detection
        
    Returns:
        DataFrame with outliers removed
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    
    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data

def normalize_minmax(data, column):
    """
    Normalize data using Min-Max scaling.
    
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
    Standardize data using Z-score normalization.
    
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

def clean_dataset(data, numeric_columns=None, outlier_multiplier=1.5):
    """
    Comprehensive data cleaning pipeline.
    
    Args:
        data: pandas DataFrame
        numeric_columns: list of numeric columns to process
        outlier_multiplier: IQR multiplier for outlier detection
        
    Returns:
        Cleaned DataFrame
    """
    if numeric_columns is None:
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_data = data.copy()
    
    for column in numeric_columns:
        if column in cleaned_data.columns:
            # Remove outliers
            cleaned_data = remove_outliers_iqr(cleaned_data, column, outlier_multiplier)
            
            # Normalize the column
            cleaned_data[f"{column}_normalized"] = normalize_minmax(cleaned_data, column)
            cleaned_data[f"{column}_standardized"] = standardize_zscore(cleaned_data, column)
    
    return cleaned_data

def validate_data(data, required_columns, numeric_threshold=0.8):
    """
    Validate data quality and completeness.
    
    Args:
        data: pandas DataFrame
        required_columns: list of required column names
        numeric_threshold: minimum proportion of numeric values required
        
    Returns:
        Dictionary with validation results
    """
    validation_results = {
        'missing_columns': [],
        'high_missing_values': [],
        'low_numeric_ratio': []
    }
    
    # Check for missing required columns
    for col in required_columns:
        if col not in data.columns:
            validation_results['missing_columns'].append(col)
    
    # Check for columns with high percentage of missing values
    for col in data.columns:
        missing_ratio = data[col].isnull().mean()
        if missing_ratio > 0.3:
            validation_results['high_missing_values'].append((col, missing_ratio))
    
    # Check numeric columns for sufficient numeric values
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        numeric_ratio = data[col].notnull().mean()
        if numeric_ratio < numeric_threshold:
            validation_results['low_numeric_ratio'].append((col, numeric_ratio))
    
    return validation_results