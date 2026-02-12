
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def normalize_minmax(df, column):
    min_val = df[column].min()
    max_val = df[column].max()
    df[column + '_normalized'] = (df[column] - min_val) / (max_val - min_val)
    return df

def clean_dataset(filepath):
    df = pd.read_csv(filepath)
    
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_columns:
        df = remove_outliers_iqr(df, col)
        df = normalize_minmax(df, col)
    
    return df

if __name__ == "__main__":
    cleaned_data = clean_dataset('raw_data.csv')
    cleaned_data.to_csv('cleaned_data.csv', index=False)
    print(f"Data cleaning complete. Original shape: {pd.read_csv('raw_data.csv').shape}, Cleaned shape: {cleaned_data.shape}")
import numpy as np

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a specified column using the IQR method.
    
    Parameters:
    data (list or np.array): The dataset
    column (int): Index of the column to clean
    
    Returns:
    np.array: Data with outliers removed
    """
    if not isinstance(data, np.ndarray):
        data = np.array(data)
    
    column_data = data[:, column].astype(float)
    
    Q1 = np.percentile(column_data, 25)
    Q3 = np.percentile(column_data, 75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    mask = (column_data >= lower_bound) & (column_data <= upper_bound)
    cleaned_data = data[mask]
    
    return cleaned_data

def calculate_statistics(data, column):
    """
    Calculate basic statistics for a column.
    
    Parameters:
    data (list or np.array): The dataset
    column (int): Index of the column to analyze
    
    Returns:
    dict: Dictionary containing statistics
    """
    if not isinstance(data, np.ndarray):
        data = np.array(data)
    
    column_data = data[:, column].astype(float)
    
    stats = {
        'mean': np.mean(column_data),
        'median': np.median(column_data),
        'std': np.std(column_data),
        'min': np.min(column_data),
        'max': np.max(column_data),
        'q1': np.percentile(column_data, 25),
        'q3': np.percentile(column_data, 75)
    }
    
    return stats

def normalize_data(data, column, method='minmax'):
    """
    Normalize data in a column.
    
    Parameters:
    data (np.array): The dataset
    column (int): Index of the column to normalize
    method (str): Normalization method ('minmax' or 'zscore')
    
    Returns:
    np.array: Normalized data
    """
    if not isinstance(data, np.ndarray):
        data = np.array(data)
    
    column_data = data[:, column].astype(float).copy()
    
    if method == 'minmax':
        min_val = np.min(column_data)
        max_val = np.max(column_data)
        if max_val - min_val != 0:
            column_data = (column_data - min_val) / (max_val - min_val)
    
    elif method == 'zscore':
        mean_val = np.mean(column_data)
        std_val = np.std(column_data)
        if std_val != 0:
            column_data = (column_data - mean_val) / std_val
    
    data[:, column] = column_data
    return data

if __name__ == "__main__":
    # Example usage
    sample_data = np.array([
        [1, 10.5, 'A'],
        [2, 12.3, 'B'],
        [3, 9.8, 'A'],
        [4, 25.1, 'C'],  # Outlier
        [5, 11.2, 'B'],
        [6, 9.5, 'A'],
        [7, 100.0, 'D'],  # Extreme outlier
        [8, 10.9, 'B']
    ])
    
    print("Original data shape:", sample_data.shape)
    
    # Remove outliers from column 1 (numeric values)
    cleaned = remove_outliers_iqr(sample_data, 1)
    print("Cleaned data shape:", cleaned.shape)
    
    # Calculate statistics
    stats = calculate_statistics(sample_data, 1)
    print("Statistics:", stats)
    
    # Normalize data
    normalized = normalize_data(sample_data.copy(), 1, method='minmax')
    print("Normalized column 1:", normalized[:, 1])