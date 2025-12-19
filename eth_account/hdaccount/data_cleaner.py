
def remove_duplicates_preserve_order(sequence):
    seen = set()
    result = []
    for item in sequence:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return resultimport numpy as np
import pandas as pd

def remove_outliers_iqr(data, column, factor=1.5):
    """
    Remove outliers using the Interquartile Range method.
    
    Args:
        data: pandas DataFrame
        column: column name to process
        factor: IQR multiplier (default 1.5)
    
    Returns:
        DataFrame with outliers removed
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    
    lower_bound = q1 - factor * iqr
    upper_bound = q3 + factor * iqr
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data

def normalize_minmax(data, column):
    """
    Normalize data to [0, 1] range using min-max scaling.
    
    Args:
        data: pandas DataFrame
        column: column name to normalize
    
    Returns:
        Series with normalized values
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    min_val = data[column].min()
    max_val = data[column].max()
    
    if min_val == max_val:
        return pd.Series([0.5] * len(data), index=data.index)
    
    normalized = (data[column] - min_val) / (max_val - min_val)
    return normalized

def standardize_zscore(data, column):
    """
    Standardize data using z-score normalization.
    
    Args:
        data: pandas DataFrame
        column: column name to standardize
    
    Returns:
        Series with standardized values
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    mean_val = data[column].mean()
    std_val = data[column].std()
    
    if std_val == 0:
        return pd.Series([0] * len(data), index=data.index)
    
    standardized = (data[column] - mean_val) / std_val
    return standardized

def clean_dataset(data, numeric_columns=None, outlier_factor=1.5):
    """
    Comprehensive data cleaning pipeline.
    
    Args:
        data: pandas DataFrame
        numeric_columns: list of numeric columns to process (default: all numeric)
        outlier_factor: IQR factor for outlier removal
    
    Returns:
        Cleaned DataFrame
    """
    cleaned_data = data.copy()
    
    if numeric_columns is None:
        numeric_columns = cleaned_data.select_dtypes(include=[np.number]).columns.tolist()
    
    for column in numeric_columns:
        if column in cleaned_data.columns:
            cleaned_data = remove_outliers_iqr(cleaned_data, column, outlier_factor)
    
    return cleaned_data

def create_sample_data():
    """
    Create sample data for testing the cleaning functions.
    
    Returns:
        DataFrame with sample data
    """
    np.random.seed(42)
    n_samples = 100
    
    data = pd.DataFrame({
        'feature_a': np.random.normal(50, 15, n_samples),
        'feature_b': np.random.exponential(10, n_samples),
        'feature_c': np.random.uniform(0, 100, n_samples),
        'category': np.random.choice(['A', 'B', 'C'], n_samples)
    })
    
    data.loc[np.random.choice(n_samples, 5, replace=False), 'feature_a'] = 200
    data.loc[np.random.choice(n_samples, 5, replace=False), 'feature_b'] = -50
    
    return data

if __name__ == "__main__":
    sample_data = create_sample_data()
    print("Original data shape:", sample_data.shape)
    print("Original data summary:")
    print(sample_data.describe())
    
    cleaned_data = clean_dataset(sample_data)
    print("\nCleaned data shape:", cleaned_data.shape)
    print("Cleaned data summary:")
    print(cleaned_data.describe())
    
    normalized_feature = normalize_minmax(cleaned_data, 'feature_a')
    standardized_feature = standardize_zscore(cleaned_data, 'feature_a')
    
    print("\nNormalized feature_a stats:")
    print(f"Min: {normalized_feature.min():.4f}, Max: {normalized_feature.max():.4f}")
    print(f"Mean: {normalized_feature.mean():.4f}, Std: {normalized_feature.std():.4f}")
    
    print("\nStandardized feature_a stats:")
    print(f"Min: {standardized_feature.min():.4f}, Max: {standardized_feature.max():.4f}")
    print(f"Mean: {standardized_feature.mean():.4f}, Std: {standardized_feature.std():.4f}")