
import numpy as np

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a pandas DataFrame column using the IQR method.
    
    Parameters:
    data (pd.DataFrame): Input DataFrame
    column (str): Column name to process
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed
    """
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data

def calculate_summary_statistics(data, column):
    """
    Calculate summary statistics for a column after outlier removal.
    
    Parameters:
    data (pd.DataFrame): Input DataFrame
    column (str): Column name to analyze
    
    Returns:
    dict: Dictionary containing summary statistics
    """
    if data.empty:
        return {}
    
    stats = {
        'mean': np.mean(data[column]),
        'median': np.median(data[column]),
        'std': np.std(data[column]),
        'min': np.min(data[column]),
        'max': np.max(data[column]),
        'count': len(data[column])
    }
    return stats

def process_dataset(data, column):
    """
    Complete pipeline for processing a dataset column.
    
    Parameters:
    data (pd.DataFrame): Input DataFrame
    column (str): Column name to process
    
    Returns:
    tuple: (cleaned_data, original_stats, cleaned_stats)
    """
    original_stats = calculate_summary_statistics(data, column)
    cleaned_data = remove_outliers_iqr(data, column)
    cleaned_stats = calculate_summary_statistics(cleaned_data, column)
    
    return cleaned_data, original_stats, cleaned_stats
import numpy as np
import pandas as pd

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

def normalize_minmax(data, column):
    """
    Normalize data using Min-Max scaling.
    
    Parameters:
    data (pd.DataFrame): Input dataframe
    column (str): Column name to normalize
    
    Returns:
    pd.Series: Normalized values
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in dataframe")
    
    min_val = data[column].min()
    max_val = data[column].max()
    
    if max_val == min_val:
        return pd.Series([0.5] * len(data), index=data.index)
    
    normalized = (data[column] - min_val) / (max_val - min_val)
    return normalized

def standardize_zscore(data, column):
    """
    Standardize data using Z-score normalization.
    
    Parameters:
    data (pd.DataFrame): Input dataframe
    column (str): Column name to standardize
    
    Returns:
    pd.Series: Standardized values
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in dataframe")
    
    mean_val = data[column].mean()
    std_val = data[column].std()
    
    if std_val == 0:
        return pd.Series([0] * len(data), index=data.index)
    
    standardized = (data[column] - mean_val) / std_val
    return standardized

def clean_missing_values(data, strategy='mean', columns=None):
    """
    Handle missing values in dataframe.
    
    Parameters:
    data (pd.DataFrame): Input dataframe
    strategy (str): Imputation strategy ('mean', 'median', 'mode', 'drop')
    columns (list): Specific columns to process, None for all numeric columns
    
    Returns:
    pd.DataFrame: Dataframe with missing values handled
    """
    cleaned_data = data.copy()
    
    if columns is None:
        numeric_cols = cleaned_data.select_dtypes(include=[np.number]).columns
        columns = list(numeric_cols)
    
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

def process_dataset(data, config):
    """
    Process dataset with multiple cleaning operations.
    
    Parameters:
    data (pd.DataFrame): Input dataframe
    config (dict): Configuration dictionary with processing steps
    
    Returns:
    pd.DataFrame: Processed dataframe
    """
    processed_data = data.copy()
    
    if 'remove_outliers' in config:
        for col in config['remove_outliers'].get('columns', []):
            multiplier = config['remove_outliers'].get('multiplier', 1.5)
            processed_data = remove_outliers_iqr(processed_data, col, multiplier)
    
    if 'normalize' in config:
        for col in config['normalize'].get('columns', []):
            method = config['normalize'].get('method', 'minmax')
            if method == 'minmax':
                processed_data[col] = normalize_minmax(processed_data, col)
            elif method == 'zscore':
                processed_data[col] = standardize_zscore(processed_data, col)
    
    if 'handle_missing' in config:
        strategy = config['handle_missing'].get('strategy', 'mean')
        columns = config['handle_missing'].get('columns', None)
        processed_data = clean_missing_values(processed_data, strategy, columns)
    
    return processed_data