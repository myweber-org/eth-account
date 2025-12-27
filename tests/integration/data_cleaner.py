import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column, threshold=1.5):
    """
    Remove outliers using IQR method.
    """
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data

def normalize_minmax(data, column):
    """
    Normalize column using min-max scaling.
    """
    min_val = data[column].min()
    max_val = data[column].max()
    if max_val == min_val:
        return data[column]
    normalized = (data[column] - min_val) / (max_val - min_val)
    return normalized

def standardize_zscore(data, column):
    """
    Standardize column using z-score normalization.
    """
    mean_val = data[column].mean()
    std_val = data[column].std()
    if std_val == 0:
        return data[column]
    standardized = (data[column] - mean_val) / std_val
    return standardized

def clean_dataset(df, numeric_columns, outlier_threshold=1.5, normalization_method='standardize'):
    """
    Clean dataset by removing outliers and applying normalization.
    """
    cleaned_df = df.copy()
    
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, col, outlier_threshold)
    
    for col in numeric_columns:
        if col in cleaned_df.columns:
            if normalization_method == 'minmax':
                cleaned_df[col] = normalize_minmax(cleaned_df, col)
            elif normalization_method == 'standardize':
                cleaned_df[col] = standardize_zscore(cleaned_df, col)
    
    return cleaned_df

def validate_data(df, required_columns, allow_nan=False):
    """
    Validate dataframe structure and content.
    """
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    if not allow_nan:
        nan_columns = df.columns[df.isnull().any()].tolist()
        if nan_columns:
            raise ValueError(f"Columns with NaN values: {nan_columns}")
    
    return True

def sample_usage():
    """
    Example usage of the data cleaning functions.
    """
    np.random.seed(42)
    sample_data = {
        'feature_a': np.random.normal(100, 15, 100),
        'feature_b': np.random.exponential(50, 100),
        'feature_c': np.random.uniform(0, 1, 100)
    }
    
    df = pd.DataFrame(sample_data)
    df.loc[10:15, 'feature_a'] = 500
    df.loc[20:25, 'feature_b'] = 1000
    
    numeric_cols = ['feature_a', 'feature_b', 'feature_c']
    
    try:
        validate_data(df, numeric_cols)
        cleaned_df = clean_dataset(df, numeric_cols, outlier_threshold=1.5, normalization_method='standardize')
        print(f"Original shape: {df.shape}")
        print(f"Cleaned shape: {cleaned_df.shape}")
        print(f"Cleaned data summary:\n{cleaned_df.describe()}")
        return cleaned_df
    except ValueError as e:
        print(f"Validation error: {e}")
        return None

if __name__ == "__main__":
    cleaned_data = sample_usage()