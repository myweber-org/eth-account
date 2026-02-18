
import numpy as np

def remove_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data
def remove_duplicates_preserve_order(iterable):
    seen = set()
    result = []
    for item in iterable:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return resultimport numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a pandas Series using the IQR method.
    Returns a cleaned Series with outliers set to NaN.
    """
    if not isinstance(data, pd.Series):
        raise TypeError("Input data must be a pandas Series")
    
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    cleaned_data = data.copy()
    cleaned_data[(data < lower_bound) | (data > upper_bound)] = np.nan
    return cleaned_data

def normalize_minmax(data):
    """
    Normalize data using min-max scaling to range [0, 1].
    Handles NaN values by ignoring them in calculation.
    """
    if not isinstance(data, pd.Series):
        data = pd.Series(data)
    
    valid_data = data.dropna()
    if len(valid_data) == 0:
        return pd.Series([np.nan] * len(data), index=data.index)
    
    data_min = valid_data.min()
    data_max = valid_data.max()
    
    if data_max == data_min:
        return pd.Series([0.5] * len(data), index=data.index)
    
    normalized = (data - data_min) / (data_max - data_min)
    return normalized

def clean_dataset(df, numeric_columns=None):
    """
    Clean a DataFrame by removing outliers and normalizing numeric columns.
    Returns a new DataFrame with cleaned data.
    """
    cleaned_df = df.copy()
    
    if numeric_columns is None:
        numeric_columns = cleaned_df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_columns:
        if col in cleaned_df.columns:
            # Remove outliers
            cleaned_col = remove_outliers_iqr(cleaned_df[col], col)
            # Normalize remaining values
            normalized_col = normalize_minmax(cleaned_col)
            cleaned_df[col] = normalized_col
    
    return cleaned_df

def calculate_statistics(data):
    """
    Calculate basic statistics for a numeric Series.
    Returns a dictionary with statistical measures.
    """
    if not isinstance(data, pd.Series):
        data = pd.Series(data)
    
    valid_data = data.dropna()
    
    stats_dict = {
        'count': len(valid_data),
        'mean': np.mean(valid_data) if len(valid_data) > 0 else np.nan,
        'std': np.std(valid_data, ddof=1) if len(valid_data) > 0 else np.nan,
        'min': np.min(valid_data) if len(valid_data) > 0 else np.nan,
        'max': np.max(valid_data) if len(valid_data) > 0 else np.nan,
        'median': np.median(valid_data) if len(valid_data) > 0 else np.nan,
        'skewness': stats.skew(valid_data) if len(valid_data) > 0 else np.nan,
        'kurtosis': stats.kurtosis(valid_data) if len(valid_data) > 0 else np.nan
    }
    
    return stats_dict

if __name__ == "__main__":
    # Example usage
    sample_data = pd.DataFrame({
        'A': np.random.normal(100, 15, 1000),
        'B': np.random.exponential(50, 1000),
        'C': np.random.uniform(0, 200, 1000)
    })
    
    print("Original data statistics:")
    for col in sample_data.columns:
        stats_result = calculate_statistics(sample_data[col])
        print(f"Column {col}: {stats_result}")
    
    cleaned_data = clean_dataset(sample_data)
    
    print("\nCleaned data statistics:")
    for col in cleaned_data.columns:
        stats_result = calculate_statistics(cleaned_data[col])
        print(f"Column {col}: {stats_result}")