import numpy as np
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
    
    if max_val == min_val:
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

def handle_missing_values(data, strategy='mean', columns=None):
    """
    Handle missing values in specified columns.
    
    Args:
        data: pandas DataFrame
        strategy: imputation strategy ('mean', 'median', 'mode', 'zero')
        columns: list of columns to process (None for all numeric columns)
    
    Returns:
        DataFrame with missing values handled
    """
    if columns is None:
        columns = data.select_dtypes(include=[np.number]).columns.tolist()
    
    result = data.copy()
    
    for col in columns:
        if col not in result.columns:
            continue
            
        if strategy == 'mean':
            fill_value = result[col].mean()
        elif strategy == 'median':
            fill_value = result[col].median()
        elif strategy == 'mode':
            fill_value = result[col].mode()[0] if not result[col].mode().empty else 0
        elif strategy == 'zero':
            fill_value = 0
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        result[col] = result[col].fillna(fill_value)
    
    return result

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
        'feature_c': np.random.uniform(0, 100, n_samples)
    })
    
    # Add some outliers
    data.loc[5, 'feature_a'] = 200
    data.loc[10, 'feature_b'] = 150
    
    # Add some missing values
    data.loc[15:20, 'feature_c'] = np.nan
    
    return data

if __name__ == "__main__":
    # Example usage
    sample_data = create_sample_data()
    print("Original data shape:", sample_data.shape)
    print("\nOriginal data summary:")
    print(sample_data.describe())
    
    # Remove outliers
    cleaned_data = remove_outliers_iqr(sample_data, 'feature_a')
    print("\nAfter outlier removal shape:", cleaned_data.shape)
    
    # Normalize a column
    normalized = normalize_minmax(cleaned_data, 'feature_b')
    print("\nNormalized feature_b stats:")
    print(f"Min: {normalized.min():.3f}, Max: {normalized.max():.3f}")
    
    # Handle missing values
    filled_data = handle_missing_values(sample_data, strategy='mean')
    print("\nMissing values after imputation:", filled_data.isnull().sum().sum())