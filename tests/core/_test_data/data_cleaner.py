import numpy as np
import pandas as pd

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
    if max_val == min_val:
        return df[column].apply(lambda x: 0.5)
    return df[column].apply(lambda x: (x - min_val) / (max_val - min_val))

def clean_dataset(df, numeric_columns):
    cleaned_df = df.copy()
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
            cleaned_df[col] = normalize_minmax(cleaned_df, col)
    return cleaned_df.reset_index(drop=True)

def validate_dataframe(df):
    required_checks = [
        lambda x: isinstance(x, pd.DataFrame),
        lambda x: not x.empty,
        lambda x: x.isnull().sum().sum() == 0
    ]
    for check in required_checks:
        if not check(df):
            return False
    return True

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'feature_a': np.random.normal(100, 15, 50),
        'feature_b': np.random.uniform(0, 1, 50),
        'category': np.random.choice(['X', 'Y', 'Z'], 50)
    })
    
    print("Original shape:", sample_data.shape)
    cleaned = clean_dataset(sample_data, ['feature_a', 'feature_b'])
    print("Cleaned shape:", cleaned.shape)
    print("Validation passed:", validate_dataframe(cleaned))
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
    
    # Handle 2D arrays
    if data.ndim == 2:
        column_data = data[:, column]
    else:
        column_data = data
    
    # Calculate IQR
    q1 = np.percentile(column_data, 25)
    q3 = np.percentile(column_data, 75)
    iqr = q3 - q1
    
    # Define outlier bounds
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    # Filter data
    if data.ndim == 2:
        mask = (column_data >= lower_bound) & (column_data <= upper_bound)
        return data[mask]
    else:
        return column_data[(column_data >= lower_bound) & (column_data <= upper_bound)]

def calculate_statistics(data):
    """
    Calculate basic statistics for the data.
    
    Parameters:
    data (np.array): Input data array
    
    Returns:
    dict: Dictionary containing statistics
    """
    if isinstance(data, list):
        data = np.array(data)
    
    stats = {
        'mean': np.mean(data),
        'median': np.median(data),
        'std': np.std(data),
        'min': np.min(data),
        'max': np.max(data),
        'count': len(data)
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
    if isinstance(data, list):
        data = np.array(data)
    
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

def example_usage():
    """Example usage of the data cleaning functions."""
    # Generate sample data with outliers
    np.random.seed(42)
    clean_data = np.random.normal(100, 15, 95)
    outlier_data = np.random.normal(200, 30, 5)
    sample_data = np.concatenate([clean_data, outlier_data])
    
    print("Original data statistics:")
    original_stats = calculate_statistics(sample_data)
    for key, value in original_stats.items():
        print(f"{key}: {value:.2f}")
    
    # Remove outliers
    cleaned_data = remove_outliers_iqr(sample_data, None)
    
    print("\nCleaned data statistics:")
    cleaned_stats = calculate_statistics(cleaned_data)
    for key, value in cleaned_stats.items():
        print(f"{key}: {value:.2f}")
    
    # Normalize data
    normalized_data = normalize_data(cleaned_data, method='zscore')
    
    print(f"\nNormalized data (first 5 values): {normalized_data[:5]}")
    
    return cleaned_data, normalized_data

if __name__ == "__main__":
    example_usage()
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range method.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        column (str): Column name to process
    
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
    
    return filtered_df.reset_index(drop=True)

def calculate_summary_statistics(df, column):
    """
    Calculate summary statistics for a column after outlier removal.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        column (str): Column name to analyze
    
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
        'count': df[column].count()
    }
    
    return stats

def process_dataframe(df, numeric_columns):
    """
    Process multiple numeric columns to remove outliers and return cleaned DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        numeric_columns (list): List of column names to process
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    for column in numeric_columns:
        if column in cleaned_df.columns and pd.api.types.is_numeric_dtype(cleaned_df[column]):
            cleaned_df = remove_outliers_iqr(cleaned_df, column)
    
    return cleaned_df

if __name__ == "__main__":
    sample_data = {
        'temperature': [22, 23, 24, 25, 26, 27, 28, 29, 30, 100],
        'humidity': [45, 46, 47, 48, 49, 50, 51, 52, 53, 54],
        'pressure': [1013, 1012, 1014, 1015, 1016, 1017, 1018, 1019, 1020, 1100]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print()
    
    cleaned_df = process_dataframe(df, ['temperature', 'pressure'])
    print("Cleaned DataFrame:")
    print(cleaned_df)
    print()
    
    for column in ['temperature', 'pressure']:
        stats = calculate_summary_statistics(cleaned_df, column)
        print(f"Statistics for {column}:")
        for key, value in stats.items():
            print(f"  {key}: {value:.2f}")
        print()