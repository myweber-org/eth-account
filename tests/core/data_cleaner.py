
import numpy as np

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a specified column using the IQR method.
    
    Parameters:
    data (list or array-like): The dataset
    column (int or str): Column index or name if using pandas DataFrame
    
    Returns:
    tuple: (cleaned_data, outliers_removed)
    """
    if isinstance(data, list):
        data_array = np.array(data)
    else:
        data_array = data
    
    q1 = np.percentile(data_array, 25)
    q3 = np.percentile(data_array, 75)
    iqr = q3 - q1
    
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    mask = (data_array >= lower_bound) & (data_array <= upper_bound)
    cleaned_data = data_array[mask]
    outliers = data_array[~mask]
    
    return cleaned_data, outliers

def calculate_statistics(data):
    """
    Calculate basic statistics for the dataset.
    
    Parameters:
    data (array-like): Input data
    
    Returns:
    dict: Dictionary containing statistics
    """
    stats = {
        'mean': np.mean(data),
        'median': np.median(data),
        'std': np.std(data),
        'min': np.min(data),
        'max': np.max(data),
        'count': len(data)
    }
    return stats

def clean_dataset(data, columns=None):
    """
    Clean dataset by removing outliers from specified columns.
    
    Parameters:
    data (array-like or dict): Input data
    columns (list): List of columns to clean
    
    Returns:
    dict: Dictionary with cleaned data and statistics
    """
    if columns is None:
        if isinstance(data, dict):
            columns = list(data.keys())
        else:
            columns = [0]
    
    result = {}
    
    for col in columns:
        if isinstance(data, dict):
            col_data = data[col]
        else:
            col_data = data[:, col] if hasattr(data, 'shape') else data
        
        cleaned, outliers = remove_outliers_iqr(col_data, col)
        stats = calculate_statistics(cleaned)
        
        result[col] = {
            'cleaned_data': cleaned,
            'outliers': outliers,
            'statistics': stats,
            'outliers_count': len(outliers)
        }
    
    return result

if __name__ == "__main__":
    # Example usage
    sample_data = np.random.normal(100, 15, 1000)
    sample_data_with_outliers = np.append(sample_data, [10, 200, 300, -50])
    
    cleaned, outliers = remove_outliers_iqr(sample_data_with_outliers, 0)
    print(f"Original data points: {len(sample_data_with_outliers)}")
    print(f"Cleaned data points: {len(cleaned)}")
    print(f"Outliers removed: {len(outliers)}")
    
    stats = calculate_statistics(cleaned)
    print(f"Statistics after cleaning: {stats}")