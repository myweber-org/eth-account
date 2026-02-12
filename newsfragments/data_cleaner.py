def remove_duplicates(input_list):
    """
    Removes duplicate elements from a list while preserving the original order.
    Returns a new list with unique elements.
    """
    seen = set()
    result = []
    for item in input_list:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column, multiplier=1.5):
    """
    Remove outliers using IQR method
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr
    
    return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]

def remove_outliers_zscore(data, column, threshold=3):
    """
    Remove outliers using Z-score method
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    z_scores = np.abs(stats.zscore(data[column].dropna()))
    mask = z_scores < threshold
    return data[mask]

def normalize_minmax(data, column):
    """
    Normalize data using Min-Max scaling
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    min_val = data[column].min()
    max_val = data[column].max()
    
    if max_val == min_val:
        return data[column].apply(lambda x: 0.5)
    
    return (data[column] - min_val) / (max_val - min_val)

def normalize_zscore(data, column):
    """
    Normalize data using Z-score standardization
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    mean_val = data[column].mean()
    std_val = data[column].std()
    
    if std_val == 0:
        return data[column].apply(lambda x: 0)
    
    return (data[column] - mean_val) / std_val

def clean_dataset(df, numeric_columns=None, outlier_method='iqr', normalize_method='minmax'):
    """
    Comprehensive data cleaning pipeline
    """
    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_df = df.copy()
    
    for col in numeric_columns:
        if col not in cleaned_df.columns:
            continue
            
        if outlier_method == 'iqr':
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
        elif outlier_method == 'zscore':
            cleaned_df = remove_outliers_zscore(cleaned_df, col)
        
        if normalize_method == 'minmax':
            cleaned_df[col] = normalize_minmax(cleaned_df, col)
        elif normalize_method == 'zscore':
            cleaned_df[col] = normalize_zscore(cleaned_df, col)
    
    return cleaned_df

def validate_data(df, required_columns=None, allow_nan_ratio=0.1):
    """
    Validate dataset structure and quality
    """
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
    
    nan_ratio = df.isnull().sum().sum() / (df.shape[0] * df.shape[1])
    if nan_ratio > allow_nan_ratio:
        raise ValueError(f"NaN ratio {nan_ratio:.2%} exceeds allowed threshold {allow_nan_ratio:.2%}")
    
    return True
import pandas as pd
import numpy as np

def clean_dataset(df, missing_strategy='mean', outlier_threshold=3):
    """
    Clean a dataset by handling missing values and removing outliers.
    
    Parameters:
    df (pd.DataFrame): Input dataframe
    missing_strategy (str): Strategy for handling missing values ('mean', 'median', 'mode', 'drop')
    outlier_threshold (float): Z-score threshold for outlier detection
    
    Returns:
    pd.DataFrame: Cleaned dataframe
    """
    cleaned_df = df.copy()
    
    # Handle missing values
    for column in cleaned_df.select_dtypes(include=[np.number]).columns:
        if cleaned_df[column].isnull().any():
            if missing_strategy == 'mean':
                cleaned_df[column].fillna(cleaned_df[column].mean(), inplace=True)
            elif missing_strategy == 'median':
                cleaned_df[column].fillna(cleaned_df[column].median(), inplace=True)
            elif missing_strategy == 'mode':
                cleaned_df[column].fillna(cleaned_df[column].mode()[0], inplace=True)
            elif missing_strategy == 'drop':
                cleaned_df.dropna(subset=[column], inplace=True)
    
    # Remove outliers using Z-score method
    if outlier_threshold > 0:
        numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
        z_scores = np.abs((cleaned_df[numeric_cols] - cleaned_df[numeric_cols].mean()) / cleaned_df[numeric_cols].std())
        outlier_mask = (z_scores < outlier_threshold).all(axis=1)
        cleaned_df = cleaned_df[outlier_mask]
    
    # Reset index after cleaning
    cleaned_df.reset_index(drop=True, inplace=True)
    
    return cleaned_df

def validate_data(df, required_columns=None, min_rows=1):
    """
    Validate dataset structure and content.
    
    Parameters:
    df (pd.DataFrame): Dataframe to validate
    required_columns (list): List of required column names
    min_rows (int): Minimum number of rows required
    
    Returns:
    tuple: (bool, str) Validation result and message
    """
    if df.empty:
        return False, "Dataframe is empty"
    
    if len(df) < min_rows:
        return False, f"Dataframe has fewer than {min_rows} rows"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    return True, "Data validation passed"

def normalize_data(df, columns=None, method='minmax'):
    """
    Normalize numerical columns in the dataframe.
    
    Parameters:
    df (pd.DataFrame): Input dataframe
    columns (list): List of columns to normalize (None for all numeric columns)
    method (str): Normalization method ('minmax' or 'zscore')
    
    Returns:
    pd.DataFrame: Dataframe with normalized columns
    """
    normalized_df = df.copy()
    
    if columns is None:
        columns = normalized_df.select_dtypes(include=[np.number]).columns
    
    for column in columns:
        if column in normalized_df.columns and pd.api.types.is_numeric_dtype(normalized_df[column]):
            if method == 'minmax':
                col_min = normalized_df[column].min()
                col_max = normalized_df[column].max()
                if col_max != col_min:
                    normalized_df[column] = (normalized_df[column] - col_min) / (col_max - col_min)
            elif method == 'zscore':
                col_mean = normalized_df[column].mean()
                col_std = normalized_df[column].std()
                if col_std != 0:
                    normalized_df[column] = (normalized_df[column] - col_mean) / col_std
    
    return normalized_df

# Example usage
if __name__ == "__main__":
    # Create sample data
    sample_data = {
        'feature1': [1, 2, np.nan, 4, 100],
        'feature2': [5, 6, 7, np.nan, 8],
        'category': ['A', 'B', 'A', 'B', 'A']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original data:")
    print(df)
    
    # Clean the data
    cleaned = clean_dataset(df, missing_strategy='mean', outlier_threshold=2)
    print("\nCleaned data:")
    print(cleaned)
    
    # Validate the cleaned data
    is_valid, message = validate_data(cleaned, required_columns=['feature1', 'feature2'])
    print(f"\nValidation: {is_valid} - {message}")
    
    # Normalize the data
    normalized = normalize_data(cleaned, method='minmax')
    print("\nNormalized data:")
    print(normalized)