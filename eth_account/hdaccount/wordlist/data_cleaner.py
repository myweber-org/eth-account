import numpy as np
import pandas as pd
from scipy import stats

def normalize_data(data, method='zscore'):
    """
    Normalize data using specified method.
    
    Args:
        data: Input data array or Series
        method: Normalization method ('zscore', 'minmax', 'robust')
    
    Returns:
        Normalized data
    """
    if method == 'zscore':
        return (data - np.mean(data)) / np.std(data)
    elif method == 'minmax':
        return (data - np.min(data)) / (np.max(data) - np.min(data))
    elif method == 'robust':
        median = np.median(data)
        q75, q25 = np.percentile(data, [75, 25])
        iqr = q75 - q25
        return (data - median) / iqr
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def remove_outliers_iqr(data, factor=1.5):
    """
    Remove outliers using IQR method.
    
    Args:
        data: Input data array
        factor: IQR multiplier for outlier detection
    
    Returns:
        Data with outliers removed
    """
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower_bound = q1 - factor * iqr
    upper_bound = q3 + factor * iqr
    
    mask = (data >= lower_bound) & (data <= upper_bound)
    return data[mask]

def detect_anomalies_zscore(data, threshold=3):
    """
    Detect anomalies using Z-score method.
    
    Args:
        data: Input data array
        threshold: Z-score threshold for anomaly detection
    
    Returns:
        Boolean mask indicating anomalies
    """
    z_scores = np.abs(stats.zscore(data))
    return z_scores > threshold

def clean_dataframe(df, columns=None, normalize=True, remove_outliers=True):
    """
    Clean DataFrame by normalizing and removing outliers.
    
    Args:
        df: Input DataFrame
        columns: Columns to clean (default: all numeric columns)
        normalize: Whether to normalize data
        remove_outliers: Whether to remove outliers
    
    Returns:
        Cleaned DataFrame
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    cleaned_df = df.copy()
    
    for col in columns:
        if col not in df.columns:
            continue
            
        col_data = df[col].dropna()
        
        if remove_outliers:
            col_data = remove_outliers_iqr(col_data)
        
        if normalize and len(col_data) > 0:
            col_data = normalize_data(col_data)
        
        cleaned_df.loc[col_data.index, col] = col_data
    
    return cleaned_df

def validate_data(data, check_missing=True, check_infinite=True):
    """
    Validate data for common issues.
    
    Args:
        data: Input data array or Series
        check_missing: Check for missing values
        check_infinite: Check for infinite values
    
    Returns:
        Dictionary with validation results
    """
    validation = {}
    
    if check_missing:
        validation['missing_count'] = np.sum(pd.isna(data))
        validation['missing_percentage'] = validation['missing_count'] / len(data) * 100
    
    if check_infinite:
        if hasattr(data, '__array__'):
            data_array = np.asarray(data)
            validation['infinite_count'] = np.sum(~np.isfinite(data_array))
    
    validation['total_count'] = len(data)
    validation['unique_count'] = len(np.unique(data))
    
    return validation

def main():
    """Example usage of data cleaning functions."""
    # Create sample data with outliers
    np.random.seed(42)
    data = np.random.normal(100, 15, 1000)
    data = np.append(data, [500, 600, 700])  # Add outliers
    
    print("Original data statistics:")
    print(f"Mean: {np.mean(data):.2f}")
    print(f"Std: {np.std(data):.2f}")
    print(f"Min: {np.min(data):.2f}")
    print(f"Max: {np.max(data):.2f}")
    
    # Clean data
    cleaned_data = remove_outliers_iqr(data)
    normalized_data = normalize_data(cleaned_data)
    
    print("\nCleaned data statistics:")
    print(f"Mean: {np.mean(normalized_data):.2f}")
    print(f"Std: {np.std(normalized_data):.2f}")
    
    # Validate data
    validation = validate_data(cleaned_data)
    print("\nData validation:")
    for key, value in validation.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    main()