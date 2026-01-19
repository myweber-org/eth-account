
import pandas as pd
import numpy as np
from scipy import stats

def remove_outliers_iqr(df, columns):
    df_clean = df.copy()
    for col in columns:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
    return df_clean

def remove_outliers_zscore(df, columns, threshold=3):
    df_clean = df.copy()
    for col in columns:
        z_scores = np.abs(stats.zscore(df_clean[col]))
        df_clean = df_clean[z_scores < threshold]
    return df_clean

def normalize_minmax(df, columns):
    df_norm = df.copy()
    for col in columns:
        min_val = df_norm[col].min()
        max_val = df_norm[col].max()
        df_norm[col] = (df_norm[col] - min_val) / (max_val - min_val)
    return df_norm

def normalize_zscore(df, columns):
    df_norm = df.copy()
    for col in columns:
        mean_val = df_norm[col].mean()
        std_val = df_norm[col].std()
        df_norm[col] = (df_norm[col] - mean_val) / std_val
    return df_norm

def clean_dataset(df, numeric_columns, outlier_method='iqr', normalize_method='minmax'):
    if outlier_method == 'iqr':
        df = remove_outliers_iqr(df, numeric_columns)
    elif outlier_method == 'zscore':
        df = remove_outliers_zscore(df, numeric_columns)
    
    if normalize_method == 'minmax':
        df = normalize_minmax(df, numeric_columns)
    elif normalize_method == 'zscore':
        df = normalize_zscore(df, numeric_columns)
    
    return df.reset_index(drop=True)
import numpy as np

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a dataset using the Interquartile Range method.
    
    Args:
        data: numpy array or list of numerical values
        column: column index or name if using structured array
        
    Returns:
        cleaned_data: data with outliers removed
        outlier_indices: indices of removed outliers
    """
    if isinstance(data, np.ndarray):
        if data.ndim > 1:
            values = data[:, column] if isinstance(column, int) else data[column]
        else:
            values = data
    else:
        values = np.array(data)
    
    q1 = np.percentile(values, 25)
    q3 = np.percentile(values, 75)
    iqr = q3 - q1
    
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    outlier_mask = (values < lower_bound) | (values > upper_bound)
    cleaned_values = values[~outlier_mask]
    outlier_indices = np.where(outlier_mask)[0]
    
    return cleaned_values, outlier_indices

def calculate_statistics(data):
    """
    Calculate basic statistics for a dataset.
    
    Args:
        data: numpy array of numerical values
        
    Returns:
        dict: dictionary containing statistical measures
    """
    stats = {
        'mean': np.mean(data),
        'median': np.median(data),
        'std': np.std(data),
        'min': np.min(data),
        'max': np.max(data),
        'q1': np.percentile(data, 25),
        'q3': np.percentile(data, 75)
    }
    return stats

def test_data_cleaner():
    """Test the outlier removal function with sample data."""
    np.random.seed(42)
    normal_data = np.random.normal(100, 15, 1000)
    outliers = np.array([200, 250, 300, 350, 400])
    test_data = np.concatenate([normal_data, outliers])
    
    cleaned_data, removed_indices = remove_outliers_iqr(test_data, 0)
    
    print(f"Original data size: {len(test_data)}")
    print(f"Cleaned data size: {len(cleaned_data)}")
    print(f"Removed {len(removed_indices)} outliers")
    print(f"Outlier indices: {removed_indices}")
    
    stats = calculate_statistics(cleaned_data)
    print("\nStatistics for cleaned data:")
    for key, value in stats.items():
        print(f"{key}: {value:.2f}")

if __name__ == "__main__":
    test_data_cleaner()