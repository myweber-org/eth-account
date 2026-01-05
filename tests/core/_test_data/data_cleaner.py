
import numpy as np

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a specified column using the Interquartile Range method.
    
    Parameters:
    data (list or np.array): Input data array
    column (int): Column index to process (for 2D arrays)
    
    Returns:
    np.array: Data with outliers removed
    """
    if isinstance(data, list):
        data = np.array(data)
    
    if data.ndim == 1:
        column_data = data
    else:
        column_data = data[:, column]
    
    q1 = np.percentile(column_data, 25)
    q3 = np.percentile(column_data, 75)
    iqr = q3 - q1
    
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    if data.ndim == 1:
        filtered_data = data[(data >= lower_bound) & (data <= upper_bound)]
    else:
        mask = (column_data >= lower_bound) & (column_data <= upper_bound)
        filtered_data = data[mask]
    
    return filtered_data

def calculate_statistics(data):
    """
    Calculate basic statistics for the data.
    
    Parameters:
    data (np.array): Input data array
    
    Returns:
    dict: Dictionary containing mean, median, std, min, max
    """
    stats = {
        'mean': np.mean(data),
        'median': np.median(data),
        'std': np.std(data),
        'min': np.min(data),
        'max': np.max(data)
    }
    return stats

def normalize_data(data, method='minmax'):
    """
    Normalize data using specified method.
    
    Parameters:
    data (np.array): Input data array
    method (str): Normalization method ('minmax' or 'zscore')
    
    Returns:
    np.array: Normalized data
    """
    if method == 'minmax':
        data_min = np.min(data)
        data_max = np.max(data)
        if data_max - data_min == 0:
            return np.zeros_like(data)
        return (data - data_min) / (data_max - data_min)
    
    elif method == 'zscore':
        data_mean = np.mean(data)
        data_std = np.std(data)
        if data_std == 0:
            return np.zeros_like(data)
        return (data - data_mean) / data_std
    
    else:
        raise ValueError("Method must be 'minmax' or 'zscore'")

def process_dataset(data, column=0, remove_outliers=True, normalize=False):
    """
    Complete data processing pipeline.
    
    Parameters:
    data: Input data
    column: Column to process (for 2D data)
    remove_outliers: Whether to remove outliers
    normalize: Whether to normalize data
    
    Returns:
    tuple: (processed_data, statistics)
    """
    if isinstance(data, list):
        data = np.array(data)
    
    original_stats = calculate_statistics(data if data.ndim == 1 else data[:, column])
    
    if remove_outliers:
        data = remove_outliers_iqr(data, column)
    
    processed_stats = calculate_statistics(data if data.ndim == 1 else data[:, column])
    
    if normalize:
        if data.ndim == 1:
            data = normalize_data(data)
        else:
            data[:, column] = normalize_data(data[:, column])
    
    return data, {
        'original': original_stats,
        'processed': processed_stats
    }