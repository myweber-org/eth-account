
import numpy as np

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a dataset using the Interquartile Range method.
    
    Parameters:
    data (numpy.ndarray): Input data array
    column (int): Column index to check for outliers
    
    Returns:
    numpy.ndarray: Data with outliers removed
    """
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a numpy array")
    
    if column >= data.shape[1]:
        raise IndexError("Column index out of bounds")
    
    q1 = np.percentile(data[:, column], 25)
    q3 = np.percentile(data[:, column], 75)
    iqr = q3 - q1
    
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    mask = (data[:, column] >= lower_bound) & (data[:, column] <= upper_bound)
    return data[mask]

def calculate_statistics(data):
    """
    Calculate basic statistics for the cleaned data.
    
    Parameters:
    data (numpy.ndarray): Input data array
    
    Returns:
    dict: Dictionary containing mean, median, and std
    """
    if data.size == 0:
        return {"mean": np.nan, "median": np.nan, "std": np.nan}
    
    return {
        "mean": np.mean(data, axis=0),
        "median": np.median(data, axis=0),
        "std": np.std(data, axis=0)
    }

def validate_data(data, expected_columns):
    """
    Validate data shape and check for NaN values.
    
    Parameters:
    data (numpy.ndarray): Input data array
    expected_columns (int): Expected number of columns
    
    Returns:
    bool: True if data is valid, False otherwise
    """
    if data.shape[1] != expected_columns:
        return False
    
    if np.any(np.isnan(data)):
        return False
    
    return True

def process_dataset(data, target_column):
    """
    Main function to process dataset by removing outliers and calculating statistics.
    
    Parameters:
    data (numpy.ndarray): Input data array
    target_column (int): Column index for outlier detection
    
    Returns:
    tuple: (cleaned_data, statistics, is_valid)
    """
    if not validate_data(data, data.shape[1]):
        raise ValueError("Invalid data format")
    
    cleaned_data = remove_outliers_iqr(data, target_column)
    stats = calculate_statistics(cleaned_data)
    
    return cleaned_data, stats, True