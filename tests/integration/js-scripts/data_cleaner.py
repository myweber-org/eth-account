
import numpy as np

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a specified column using the IQR method.
    
    Args:
        data (list or np.array): Input data
        column (int): Column index for 2D data, or None for 1D data
    
    Returns:
        np.array: Data with outliers removed
    """
    if column is not None:
        column_data = np.array(data)[:, column].astype(float)
    else:
        column_data = np.array(data).astype(float)
    
    q1 = np.percentile(column_data, 25)
    q3 = np.percentile(column_data, 75)
    iqr = q3 - q1
    
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    if column is not None:
        filtered_indices = np.where((column_data >= lower_bound) & (column_data <= upper_bound))[0]
        return np.array(data)[filtered_indices]
    else:
        return column_data[(column_data >= lower_bound) & (column_data <= upper_bound)]

def calculate_statistics(data):
    """
    Calculate basic statistics for the data.
    
    Args:
        data (list or np.array): Input data
    
    Returns:
        dict: Dictionary containing mean, median, std, min, max
    """
    data_array = np.array(data).astype(float)
    
    stats = {
        'mean': np.mean(data_array),
        'median': np.median(data_array),
        'std': np.std(data_array),
        'min': np.min(data_array),
        'max': np.max(data_array)
    }
    
    return stats

def clean_dataset(data, column=None):
    """
    Main function to clean dataset by removing outliers.
    
    Args:
        data (list or np.array): Input data
        column (int, optional): Column index for 2D data
    
    Returns:
        tuple: (cleaned_data, removed_count, statistics)
    """
    original_length = len(data)
    cleaned_data = remove_outliers_iqr(data, column)
    removed_count = original_length - len(cleaned_data)
    
    if column is not None:
        stats_data = np.array(cleaned_data)[:, column].astype(float)
    else:
        stats_data = cleaned_data
    
    statistics = calculate_statistics(stats_data)
    
    return cleaned_data, removed_count, statistics