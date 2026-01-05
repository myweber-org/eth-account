
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
import numpy as np
import pandas as pd

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to clean
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
    return filtered_df

def calculate_summary_stats(df, column):
    """
    Calculate summary statistics for a column.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name
    
    Returns:
    dict: Dictionary containing summary statistics
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    stats = {
        'mean': df[column].mean(),
        'median': df[column].median(),
        'std': df[column].std(),
        'min': df[column].min(),
        'max': df[column].max(),
        'count': df[column].count(),
        'missing': df[column].isnull().sum()
    }
    
    return stats

def clean_numeric_data(df, columns=None):
    """
    Clean numeric data by removing outliers from specified columns.
    If no columns specified, clean all numeric columns.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    columns (list): List of column names to clean
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    if columns is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        columns = numeric_cols
    
    cleaned_df = df.copy()
    
    for column in columns:
        if column in cleaned_df.columns:
            if pd.api.types.is_numeric_dtype(cleaned_df[column]):
                original_count = len(cleaned_df)
                cleaned_df = remove_outliers_iqr(cleaned_df, column)
                removed_count = original_count - len(cleaned_df)
                print(f"Removed {removed_count} outliers from column '{column}'")
    
    return cleaned_df

if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    
    # Create sample data with outliers
    data = {
        'A': np.random.normal(100, 15, 1000),
        'B': np.random.exponential(50, 1000),
        'C': np.random.uniform(0, 200, 1000)
    }
    
    # Add some outliers
    data['A'][:50] = np.random.normal(300, 10, 50)
    data['B'][:30] = np.random.normal(500, 20, 30)
    
    df = pd.DataFrame(data)
    
    print("Original DataFrame shape:", df.shape)
    print("\nOriginal summary statistics for column 'A':")
    print(calculate_summary_stats(df, 'A'))
    
    # Clean the data
    cleaned_df = clean_numeric_data(df, ['A', 'B'])
    
    print("\nCleaned DataFrame shape:", cleaned_df.shape)
    print("\nCleaned summary statistics for column 'A':")
    print(calculate_summary_stats(cleaned_df, 'A'))