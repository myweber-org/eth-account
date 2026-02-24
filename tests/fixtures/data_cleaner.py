import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column, factor=1.5):
    """
    Remove outliers from a pandas Series using the IQR method.
    
    Args:
        data: pandas Series containing the data
        column: Name of the column to process
        factor: IQR multiplier for outlier detection (default: 1.5)
    
    Returns:
        pandas Series with outliers removed
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in data")
    
    series = data[column]
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    
    lower_bound = q1 - factor * iqr
    upper_bound = q3 + factor * iqr
    
    return data[(series >= lower_bound) & (series <= upper_bound)]

def normalize_minmax(data, column):
    """
    Normalize data to [0, 1] range using min-max scaling.
    
    Args:
        data: pandas Series containing the data
        column: Name of the column to normalize
    
    Returns:
        pandas Series with normalized values
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in data")
    
    series = data[column]
    min_val = series.min()
    max_val = series.max()
    
    if max_val == min_val:
        return pd.Series([0.5] * len(series), index=series.index)
    
    return (series - min_val) / (max_val - min_val)

def standardize_zscore(data, column):
    """
    Standardize data using z-score normalization (mean=0, std=1).
    
    Args:
        data: pandas Series containing the data
        column: Name of the column to standardize
    
    Returns:
        pandas Series with standardized values
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in data")
    
    series = data[column]
    mean = series.mean()
    std = series.std()
    
    if std == 0:
        return pd.Series([0] * len(series), index=series.index)
    
    return (series - mean) / std

def clean_dataset(df, numeric_columns=None, outlier_factor=1.5, normalize=True):
    """
    Comprehensive data cleaning pipeline for numeric columns.
    
    Args:
        df: Input pandas DataFrame
        numeric_columns: List of numeric column names to process (default: all numeric)
        outlier_factor: IQR multiplier for outlier removal
        normalize: Whether to apply min-max normalization after outlier removal
    
    Returns:
        Cleaned pandas DataFrame
    """
    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_df = df.copy()
    
    for col in numeric_columns:
        if col not in cleaned_df.columns:
            continue
            
        # Remove outliers
        cleaned_df = remove_outliers_iqr(cleaned_df, col, outlier_factor)
        
        # Normalize if requested
        if normalize:
            cleaned_df[col] = normalize_minmax(cleaned_df, col)
    
    return cleaned_df.reset_index(drop=True)

def calculate_summary_statistics(df, column):
    """
    Calculate comprehensive summary statistics for a column.
    
    Args:
        df: pandas DataFrame
        column: Name of the column to analyze
    
    Returns:
        Dictionary containing summary statistics
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in data")
    
    series = df[column]
    
    stats_dict = {
        'count': len(series),
        'mean': series.mean(),
        'median': series.median(),
        'std': series.std(),
        'min': series.min(),
        'max': series.max(),
        'q1': series.quantile(0.25),
        'q3': series.quantile(0.75),
        'skewness': series.skew(),
        'kurtosis': series.kurtosis()
    }
    
    return stats_dict

# Example usage
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    sample_data = pd.DataFrame({
        'feature_a': np.random.normal(100, 15, 1000),
        'feature_b': np.random.exponential(50, 1000),
        'category': np.random.choice(['A', 'B', 'C'], 1000)
    })
    
    # Add some outliers
    sample_data.loc[10, 'feature_a'] = 500
    sample_data.loc[20, 'feature_b'] = 1000
    
    print("Original data shape:", sample_data.shape)
    print("\nOriginal summary for feature_a:")
    print(calculate_summary_statistics(sample_data, 'feature_a'))
    
    # Clean the data
    cleaned_data = clean_dataset(sample_data, ['feature_a', 'feature_b'])
    
    print("\nCleaned data shape:", cleaned_data.shape)
    print("\nCleaned summary for feature_a:")
    print(calculate_summary_statistics(cleaned_data, 'feature_a'))
import pandas as pd

def clean_dataset(df, remove_duplicates=True, fill_missing=None):
    """
    Clean a pandas DataFrame by handling missing values and removing duplicates.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    remove_duplicates (bool): If True, remove duplicate rows.
    fill_missing (str or None): Method to fill missing values: 
        'mean', 'median', 'mode', or None to drop rows with missing values.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    # Handle missing values
    if fill_missing is not None:
        if fill_missing == 'mean':
            cleaned_df = cleaned_df.fillna(cleaned_df.mean(numeric_only=True))
        elif fill_missing == 'median':
            cleaned_df = cleaned_df.fillna(cleaned_df.median(numeric_only=True))
        elif fill_missing == 'mode':
            cleaned_df = cleaned_df.fillna(cleaned_df.mode().iloc[0])
    else:
        cleaned_df = cleaned_df.dropna()
    
    # Remove duplicates
    if remove_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    # Reset index after cleaning
    cleaned_df = cleaned_df.reset_index(drop=True)
    
    return cleaned_df

def validate_dataset(df, required_columns=None):
    """
    Validate a DataFrame for required columns and basic integrity.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of column names that must be present.
    
    Returns:
    tuple: (is_valid, error_message)
    """
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
    
    if df.empty:
        return False, "DataFrame is empty"
    
    return True, "Dataset is valid"

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'A': [1, 2, None, 4, 2],
        'B': [5, None, 7, 8, 5],
        'C': ['x', 'y', 'z', 'x', 'y']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nCleaned DataFrame (fill with mean, remove duplicates):")
    cleaned = clean_dataset(df, fill_missing='mean')
    print(cleaned)
    
    is_valid, message = validate_dataset(cleaned, required_columns=['A', 'B', 'C'])
    print(f"\nValidation: {message}")