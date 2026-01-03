import numpy as np

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a specified column using the IQR method.
    
    Parameters:
    data (list or array-like): The dataset.
    column (int or str): Column index or name if data is structured.
    
    Returns:
    cleaned_data: Data with outliers removed.
    """
    if isinstance(data, list):
        data = np.array(data)
    
    if isinstance(column, str):
        raise ValueError("Column name not supported for simple arrays. Use index.")
    
    column_data = data[:, column] if data.ndim > 1 else data
    
    q1 = np.percentile(column_data, 25)
    q3 = np.percentile(column_data, 75)
    iqr = q3 - q1
    
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    if data.ndim == 1:
        mask = (column_data >= lower_bound) & (column_data <= upper_bound)
        cleaned_data = data[mask]
    else:
        mask = (column_data >= lower_bound) & (column_data <= upper_bound)
        cleaned_data = data[mask]
    
    return cleaned_data

def example_usage():
    sample_data = np.random.randn(100, 3)
    sample_data[0, 1] = 100  # Add an outlier
    
    print("Original shape:", sample_data.shape)
    cleaned = remove_outliers_iqr(sample_data, 1)
    print("Cleaned shape:", cleaned.shape)
    
    return cleaned

if __name__ == "__main__":
    example_usage()