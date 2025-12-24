
import numpy as np

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a specified column using the IQR method.
    
    Args:
        data (list or np.array): Input data
        column (int or str): Column index or name if data is structured
        
    Returns:
        np.array: Data with outliers removed
    """
    if isinstance(data, list):
        data = np.array(data)
    
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    filtered_data = data[(data >= lower_bound) & (data <= upper_bound)]
    
    return filtered_data

def calculate_statistics(data):
    """
    Calculate basic statistics for the data.
    
    Args:
        data (np.array): Input data
        
    Returns:
        dict: Dictionary containing mean, median, and standard deviation
    """
    stats = {
        'mean': np.mean(data),
        'median': np.median(data),
        'std': np.std(data),
        'min': np.min(data),
        'max': np.max(data)
    }
    
    return stats

def clean_dataset(data, columns=None):
    """
    Clean entire dataset by removing outliers from specified columns.
    
    Args:
        data (np.array): 2D array of data
        columns (list): List of column indices to clean
        
    Returns:
        np.array: Cleaned dataset
    """
    if columns is None:
        columns = range(data.shape[1])
    
    cleaned_data = data.copy()
    
    for col in columns:
        col_data = data[:, col]
        cleaned_col = remove_outliers_iqr(col_data, col)
        
        # For simplicity, pad with mean if lengths don't match
        if len(cleaned_col) < len(col_data):
            mean_val = np.mean(cleaned_col)
            padding = np.full(len(col_data) - len(cleaned_col), mean_val)
            cleaned_col = np.concatenate([cleaned_col, padding])
        
        cleaned_data[:, col] = cleaned_col
    
    return cleaned_data

if __name__ == "__main__":
    # Example usage
    sample_data = np.random.randn(100, 3) * 10 + 50
    print("Original data shape:", sample_data.shape)
    
    cleaned = clean_dataset(sample_data)
    print("Cleaned data shape:", cleaned.shape)
    
    stats = calculate_statistics(cleaned[:, 0])
    print("Column 1 statistics:", stats)