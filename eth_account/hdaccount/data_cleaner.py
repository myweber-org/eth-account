
import numpy as np

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a specified column using the Interquartile Range method.
    
    Parameters:
    data (numpy.ndarray): Input data array.
    column (int): Index of the column to process.
    
    Returns:
    numpy.ndarray: Data with outliers removed from specified column.
    """
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a numpy array")
    
    if column >= data.shape[1]:
        raise IndexError("Column index out of bounds")
    
    col_data = data[:, column]
    
    q1 = np.percentile(col_data, 25)
    q3 = np.percentile(col_data, 75)
    iqr = q3 - q1
    
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    mask = (col_data >= lower_bound) & (col_data <= upper_bound)
    
    return data[mask]

def calculate_statistics(data, column):
    """
    Calculate basic statistics for a specified column.
    
    Parameters:
    data (numpy.ndarray): Input data array.
    column (int): Index of the column to analyze.
    
    Returns:
    dict: Dictionary containing statistical measures.
    """
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a numpy array")
    
    col_data = data[:, column]
    
    stats = {
        'mean': np.mean(col_data),
        'median': np.median(col_data),
        'std': np.std(col_data),
        'min': np.min(col_data),
        'max': np.max(col_data),
        'q1': np.percentile(col_data, 25),
        'q3': np.percentile(col_data, 75)
    }
    
    return stats

def validate_data_shape(data, expected_columns):
    """
    Validate that data has the expected number of columns.
    
    Parameters:
    data (numpy.ndarray): Input data array.
    expected_columns (int): Expected number of columns.
    
    Returns:
    bool: True if shape is valid, False otherwise.
    """
    if not isinstance(data, np.ndarray):
        return False
    
    return data.shape[1] == expected_columns

def example_usage():
    """
    Example demonstrating how to use the data cleaning functions.
    """
    np.random.seed(42)
    
    sample_data = np.random.randn(100, 3)
    sample_data[0, 0] = 100
    
    print("Original data shape:", sample_data.shape)
    
    cleaned_data = remove_outliers_iqr(sample_data, 0)
    print("Cleaned data shape:", cleaned_data.shape)
    
    stats = calculate_statistics(cleaned_data, 0)
    print("Statistics for column 0:", stats)
    
    is_valid = validate_data_shape(cleaned_data, 3)
    print("Data shape validation:", is_valid)

if __name__ == "__main__":
    example_usage()