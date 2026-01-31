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
    main()def remove_duplicates(input_list):
    """
    Removes duplicate items from a list while preserving the original order.
    """
    seen = set()
    result = []
    for item in input_list:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return resultimport pandas as pd
import numpy as np

def clean_missing_values(df, strategy='mean', columns=None):
    """
    Handle missing values in a DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        strategy (str): Strategy for imputation ('mean', 'median', 'mode', 'drop')
        columns (list): Specific columns to clean, if None clean all columns
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    df_clean = df.copy()
    
    if columns is None:
        columns = df_clean.columns
    
    for col in columns:
        if df_clean[col].isnull().any():
            if strategy == 'mean':
                df_clean[col].fillna(df_clean[col].mean(), inplace=True)
            elif strategy == 'median':
                df_clean[col].fillna(df_clean[col].median(), inplace=True)
            elif strategy == 'mode':
                df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)
            elif strategy == 'drop':
                df_clean.dropna(subset=[col], inplace=True)
    
    return df_clean

def remove_outliers_iqr(df, columns=None, multiplier=1.5):
    """
    Remove outliers using the Interquartile Range method.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        columns (list): Columns to check for outliers
        multiplier (float): IQR multiplier for outlier detection
    
    Returns:
        pd.DataFrame: DataFrame with outliers removed
    """
    df_clean = df.copy()
    
    if columns is None:
        columns = df_clean.select_dtypes(include=[np.number]).columns
    
    for col in columns:
        if col in df_clean.columns:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - multiplier * IQR
            upper_bound = Q3 + multiplier * IQR
            
            df_clean = df_clean[(df_clean[col] >= lower_bound) & 
                               (df_clean[col] <= upper_bound)]
    
    return df_clean

def standardize_data(df, columns=None):
    """
    Standardize numerical columns to have zero mean and unit variance.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        columns (list): Columns to standardize
    
    Returns:
        pd.DataFrame: Standardized DataFrame
    """
    from sklearn.preprocessing import StandardScaler
    
    df_clean = df.copy()
    
    if columns is None:
        columns = df_clean.select_dtypes(include=[np.number]).columns
    
    scaler = StandardScaler()
    df_clean[columns] = scaler.fit_transform(df_clean[columns])
    
    return df_clean

def clean_dataset(df, missing_strategy='mean', remove_outliers=True, standardize=False):
    """
    Comprehensive data cleaning pipeline.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        missing_strategy (str): Strategy for handling missing values
        remove_outliers (bool): Whether to remove outliers
        standardize (bool): Whether to standardize numerical columns
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    df_clean = df.copy()
    
    df_clean = clean_missing_values(df_clean, strategy=missing_strategy)
    
    if remove_outliers:
        df_clean = remove_outliers_iqr(df_clean)
    
    if standardize:
        df_clean = standardize_data(df_clean)
    
    return df_clean