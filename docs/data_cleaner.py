import numpy as np
import pandas as pd
from scipy import stats

def normalize_data(data, method='zscore'):
    """
    Normalize data using specified method.
    
    Args:
        data: numpy array or pandas Series
        method: 'zscore', 'minmax', or 'robust'
    
    Returns:
        Normalized data
    """
    if method == 'zscore':
        return (data - np.mean(data)) / np.std(data)
    elif method == 'minmax':
        return (data - np.min(data)) / (np.max(data) - np.min(data))
    elif method == 'robust':
        return (data - np.median(data)) / stats.iqr(data)
    else:
        raise ValueError("Method must be 'zscore', 'minmax', or 'robust'")

def remove_outliers_iqr(data, multiplier=1.5):
    """
    Remove outliers using IQR method.
    
    Args:
        data: numpy array or pandas Series
        multiplier: IQR multiplier for outlier detection
    
    Returns:
        Data with outliers removed
    """
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr
    
    return data[(data >= lower_bound) & (data <= upper_bound)]

def clean_dataset(df, columns=None, outlier_method='iqr'):
    """
    Clean entire dataset by normalizing and removing outliers.
    
    Args:
        df: pandas DataFrame
        columns: list of columns to clean (default: all numeric columns)
        outlier_method: 'iqr' or 'zscore' for outlier removal
    
    Returns:
        Cleaned DataFrame
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    cleaned_df = df.copy()
    
    for col in columns:
        if outlier_method == 'iqr':
            cleaned_df[col] = remove_outliers_iqr(df[col])
        elif outlier_method == 'zscore':
            z_scores = np.abs(stats.zscore(df[col].dropna()))
            cleaned_df[col] = df[col][z_scores < 3]
        
        # Normalize the cleaned data
        if cleaned_df[col].notna().any():
            cleaned_df[col] = normalize_data(cleaned_df[col].dropna())
    
    return cleaned_df

def validate_data(data, check_nan=True, check_inf=True):
    """
    Validate data for common issues.
    
    Args:
        data: numpy array or pandas Series
        check_nan: check for NaN values
        check_inf: check for infinite values
    
    Returns:
        Dictionary with validation results
    """
    validation = {
        'has_nan': False,
        'has_inf': False,
        'is_empty': False,
        'data_type': str(type(data))
    }
    
    if len(data) == 0:
        validation['is_empty'] = True
        return validation
    
    if check_nan:
        validation['has_nan'] = np.any(pd.isna(data))
    
    if check_inf:
        validation['has_inf'] = np.any(np.isinf(data))
    
    return validation

# Example usage
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    sample_data = np.random.normal(100, 15, 1000)
    
    # Add some outliers
    sample_data = np.append(sample_data, [10, 250, 300])
    
    # Create DataFrame
    df = pd.DataFrame({
        'values': sample_data,
        'category': np.random.choice(['A', 'B', 'C'], len(sample_data))
    })
    
    # Clean the data
    cleaned = clean_dataset(df, columns=['values'])
    
    # Validate
    validation = validate_data(cleaned['values'])
    print(f"Validation results: {validation}")
    print(f"Original shape: {df.shape}, Cleaned shape: {cleaned.shape}")