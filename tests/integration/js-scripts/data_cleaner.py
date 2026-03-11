
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column, factor=1.5):
    """
    Remove outliers using Interquartile Range method.
    
    Parameters:
    data (pd.DataFrame): Input dataframe
    column (str): Column name to process
    factor (float): Multiplier for IQR
    
    Returns:
    pd.DataFrame: Dataframe with outliers removed
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in dataframe")
    
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - factor * iqr
    upper_bound = q3 + factor * iqr
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data

def z_score_normalize(data, column):
    """
    Normalize data using Z-score method.
    
    Parameters:
    data (pd.DataFrame): Input dataframe
    column (str): Column name to normalize
    
    Returns:
    pd.Series: Normalized column values
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in dataframe")
    
    mean_val = data[column].mean()
    std_val = data[column].std()
    
    if std_val == 0:
        return pd.Series([0] * len(data), index=data.index)
    
    normalized = (data[column] - mean_val) / std_val
    return normalized

def min_max_normalize(data, column, feature_range=(0, 1)):
    """
    Normalize data using Min-Max scaling.
    
    Parameters:
    data (pd.DataFrame): Input dataframe
    column (str): Column name to normalize
    feature_range (tuple): Desired range of transformed data
    
    Returns:
    pd.Series: Normalized column values
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in dataframe")
    
    min_val = data[column].min()
    max_val = data[column].max()
    
    if min_val == max_val:
        return pd.Series([feature_range[0]] * len(data), index=data.index)
    
    normalized = (data[column] - min_val) / (max_val - min_val)
    normalized = normalized * (feature_range[1] - feature_range[0]) + feature_range[0]
    return normalized

def clean_dataset(data, numeric_columns, outlier_method='iqr', normalize_method='zscore'):
    """
    Comprehensive data cleaning pipeline.
    
    Parameters:
    data (pd.DataFrame): Input dataframe
    numeric_columns (list): List of numeric column names to process
    outlier_method (str): 'iqr' or None for outlier removal
    normalize_method (str): 'zscore', 'minmax', or None for normalization
    
    Returns:
    pd.DataFrame: Cleaned dataframe
    """
    cleaned_data = data.copy()
    
    for column in numeric_columns:
        if column not in cleaned_data.columns:
            continue
            
        if outlier_method == 'iqr':
            cleaned_data = remove_outliers_iqr(cleaned_data, column)
        
        if normalize_method == 'zscore':
            cleaned_data[column] = z_score_normalize(cleaned_data, column)
        elif normalize_method == 'minmax':
            cleaned_data[column] = min_max_normalize(cleaned_data, column)
    
    return cleaned_data

def validate_dataframe(data, required_columns=None):
    """
    Validate dataframe structure and content.
    
    Parameters:
    data (pd.DataFrame): Dataframe to validate
    required_columns (list): List of required column names
    
    Returns:
    dict: Validation results with status and messages
    """
    validation_result = {
        'is_valid': True,
        'messages': [],
        'missing_columns': []
    }
    
    if not isinstance(data, pd.DataFrame):
        validation_result['is_valid'] = False
        validation_result['messages'].append('Input is not a pandas DataFrame')
        return validation_result
    
    if data.empty:
        validation_result['is_valid'] = False
        validation_result['messages'].append('DataFrame is empty')
    
    if required_columns:
        missing = [col for col in required_columns if col not in data.columns]
        if missing:
            validation_result['is_valid'] = False
            validation_result['missing_columns'] = missing
            validation_result['messages'].append(f'Missing required columns: {missing}')
    
    return validation_result