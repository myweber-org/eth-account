import pandas as pd
import numpy as np

def clean_missing_data(df, strategy='mean', columns=None):
    """
    Clean missing data in a DataFrame using specified strategy.
    
    Args:
        df: pandas DataFrame containing data with potential missing values
        strategy: Method for handling missing values ('mean', 'median', 'mode', 'drop')
        columns: List of column names to apply cleaning to (None for all columns)
    
    Returns:
        Cleaned pandas DataFrame
    """
    if df.empty:
        return df
    
    if columns is None:
        columns = df.columns
    
    df_clean = df.copy()
    
    for col in columns:
        if col not in df_clean.columns:
            continue
            
        if df_clean[col].isnull().any():
            if strategy == 'mean':
                df_clean[col].fillna(df_clean[col].mean(), inplace=True)
            elif strategy == 'median':
                df_clean[col].fillna(df_clean[col].median(), inplace=True)
            elif strategy == 'mode':
                df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)
            elif strategy == 'drop':
                df_clean = df_clean.dropna(subset=[col])
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
    
    return df_clean

def detect_outliers_iqr(df, column, threshold=1.5):
    """
    Detect outliers using Interquartile Range method.
    
    Args:
        df: pandas DataFrame
        column: Column name to check for outliers
        threshold: IQR multiplier threshold
    
    Returns:
        Boolean mask indicating outliers
    """
    if column not in df.columns:
        return pd.Series([False] * len(df))
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR
    
    return (df[column] < lower_bound) | (df[column] > upper_bound)

def normalize_column(df, column, method='minmax'):
    """
    Normalize a column using specified method.
    
    Args:
        df: pandas DataFrame
        column: Column name to normalize
        method: Normalization method ('minmax' or 'zscore')
    
    Returns:
        Series with normalized values
    """
    if column not in df.columns:
        return pd.Series()
    
    if method == 'minmax':
        min_val = df[column].min()
        max_val = df[column].max()
        if max_val == min_val:
            return pd.Series([0.5] * len(df))
        return (df[column] - min_val) / (max_val - min_val)
    
    elif method == 'zscore':
        mean_val = df[column].mean()
        std_val = df[column].std()
        if std_val == 0:
            return pd.Series([0] * len(df))
        return (df[column] - mean_val) / std_val
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Args:
        df: pandas DataFrame to validate
        required_columns: List of column names that must be present
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not isinstance(df, pd.DataFrame):
        return False, "Input is not a pandas DataFrame"
    
    if df.empty:
        return False, "DataFrame is empty"
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
    
    return True, "DataFrame is valid"

def example_usage():
    """Example usage of the data cleaning utilities."""
    data = {
        'temperature': [22.5, np.nan, 25.0, 30.2, 18.7, np.nan, 27.8],
        'humidity': [45, 50, np.nan, 65, 40, 55, 60],
        'pressure': [1013, 1012, 1015, 1011, 1014, 1013, 1012]
    }
    
    df = pd.DataFrame(data)
    print("Original DataFrame:")
    print(df)
    
    df_cleaned = clean_missing_data(df, strategy='mean')
    print("\nCleaned DataFrame (mean imputation):")
    print(df_cleaned)
    
    outliers = detect_outliers_iqr(df_cleaned, 'temperature')
    print(f"\nOutliers in temperature: {outliers.sum()}")
    
    df_cleaned['temperature_normalized'] = normalize_column(df_cleaned, 'temperature', 'minmax')
    print("\nDataFrame with normalized temperature:")
    print(df_cleaned)
    
    is_valid, message = validate_dataframe(df_cleaned, ['temperature', 'humidity', 'pressure'])
    print(f"\nValidation result: {is_valid}, Message: {message}")

if __name__ == "__main__":
    example_usage()