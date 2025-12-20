import numpy as np

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a specified column using the IQR method.
    
    Parameters:
    data (numpy.ndarray): Input data array
    column (int): Column index to process
    
    Returns:
    numpy.ndarray: Data with outliers removed
    """
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a numpy array")
    
    if column >= data.shape[1]:
        raise IndexError("Column index out of bounds")
    
    column_data = data[:, column]
    
    Q1 = np.percentile(column_data, 25)
    Q3 = np.percentile(column_data, 75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    mask = (column_data >= lower_bound) & (column_data <= upper_bound)
    
    return data[mask]

def calculate_statistics(data):
    """
    Calculate basic statistics for the data.
    
    Parameters:
    data (numpy.ndarray): Input data array
    
    Returns:
    dict: Dictionary containing statistics
    """
    stats = {
        'mean': np.mean(data, axis=0),
        'median': np.median(data, axis=0),
        'std': np.std(data, axis=0),
        'min': np.min(data, axis=0),
        'max': np.max(data, axis=0)
    }
    return stats

def normalize_data(data, method='minmax'):
    """
    Normalize data using specified method.
    
    Parameters:
    data (numpy.ndarray): Input data array
    method (str): Normalization method ('minmax' or 'zscore')
    
    Returns:
    numpy.ndarray: Normalized data
    """
    if method == 'minmax':
        data_min = np.min(data, axis=0)
        data_max = np.max(data, axis=0)
        return (data - data_min) / (data_max - data_min + 1e-8)
    
    elif method == 'zscore':
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        return (data - mean) / (std + 1e-8)
    
    else:
        raise ValueError("Method must be 'minmax' or 'zscore'")

def validate_data(data):
    """
    Validate data for common issues.
    
    Parameters:
    data (numpy.ndarray): Input data array
    
    Returns:
    dict: Validation results
    """
    validation = {
        'has_nan': np.any(np.isnan(data)),
        'has_inf': np.any(np.isinf(data)),
        'shape': data.shape,
        'dtype': str(data.dtype),
        'is_empty': data.size == 0
    }
    return validation

if __name__ == "__main__":
    sample_data = np.random.randn(100, 3)
    sample_data[10, 1] = 100
    
    print("Original shape:", sample_data.shape)
    print("Validation:", validate_data(sample_data))
    
    cleaned_data = remove_outliers_iqr(sample_data, 1)
    print("Cleaned shape:", cleaned_data.shape)
    
    normalized = normalize_data(cleaned_data, method='zscore')
    stats = calculate_statistics(normalized)
    print("Statistics:", {k: v.round(3) for k, v in stats.items()})