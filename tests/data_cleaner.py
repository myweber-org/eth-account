
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column, multiplier=1.5):
    """
    Remove outliers using the Interquartile Range method.
    
    Parameters:
    data (pd.DataFrame): Input dataframe
    column (str): Column name to process
    multiplier (float): IQR multiplier for outlier detection
    
    Returns:
    pd.DataFrame: Dataframe with outliers removed
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in dataframe")
    
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    
    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data.copy()

def z_score_normalization(data, column):
    """
    Apply Z-score normalization to a column.
    
    Parameters:
    data (pd.DataFrame): Input dataframe
    column (str): Column name to normalize
    
    Returns:
    pd.DataFrame: Dataframe with normalized column
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in dataframe")
    
    normalized_data = data.copy()
    mean_val = normalized_data[column].mean()
    std_val = normalized_data[column].std()
    
    if std_val > 0:
        normalized_data[f'{column}_normalized'] = (normalized_data[column] - mean_val) / std_val
    else:
        normalized_data[f'{column}_normalized'] = 0
    
    return normalized_data

def min_max_scaling(data, column, feature_range=(0, 1)):
    """
    Apply Min-Max scaling to a column.
    
    Parameters:
    data (pd.DataFrame): Input dataframe
    column (str): Column name to scale
    feature_range (tuple): Desired range of transformed data
    
    Returns:
    pd.DataFrame: Dataframe with scaled column
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in dataframe")
    
    scaled_data = data.copy()
    min_val = scaled_data[column].min()
    max_val = scaled_data[column].max()
    
    if max_val > min_val:
        scaled_data[f'{column}_scaled'] = (scaled_data[column] - min_val) / (max_val - min_val)
        scaled_data[f'{column}_scaled'] = scaled_data[f'{column}_scaled'] * (feature_range[1] - feature_range[0]) + feature_range[0]
    else:
        scaled_data[f'{column}_scaled'] = feature_range[0]
    
    return scaled_data

def handle_missing_values(data, strategy='mean', columns=None):
    """
    Handle missing values in specified columns.
    
    Parameters:
    data (pd.DataFrame): Input dataframe
    strategy (str): Imputation strategy ('mean', 'median', 'mode', 'drop')
    columns (list): List of columns to process, None for all columns
    
    Returns:
    pd.DataFrame: Dataframe with handled missing values
    """
    processed_data = data.copy()
    
    if columns is None:
        columns = processed_data.columns
    
    for column in columns:
        if column not in processed_data.columns:
            continue
            
        if processed_data[column].isnull().any():
            if strategy == 'mean':
                fill_value = processed_data[column].mean()
            elif strategy == 'median':
                fill_value = processed_data[column].median()
            elif strategy == 'mode':
                fill_value = processed_data[column].mode()[0] if not processed_data[column].mode().empty else 0
            elif strategy == 'drop':
                processed_data = processed_data.dropna(subset=[column])
                continue
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
            
            processed_data[column] = processed_data[column].fillna(fill_value)
    
    return processed_data

def validate_dataframe(data, required_columns=None, min_rows=1):
    """
    Validate dataframe structure and content.
    
    Parameters:
    data (pd.DataFrame): Dataframe to validate
    required_columns (list): List of required column names
    min_rows (int): Minimum number of rows required
    
    Returns:
    bool: True if validation passes, raises exception otherwise
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    
    if len(data) < min_rows:
        raise ValueError(f"Dataframe must have at least {min_rows} rows")
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
    
    return True