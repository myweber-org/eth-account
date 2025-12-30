
import numpy as np
import pandas as pd

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to process
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
    return filtered_df.reset_index(drop=True)

def calculate_summary_statistics(df, column):
    """
    Calculate summary statistics for a column.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to analyze
    
    Returns:
    dict: Dictionary containing summary statistics
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    stats = {
        'mean': df[column].mean(),
        'median': df[column].median(),
        'std': df[column].std(),
        'min': df[column].min(),
        'max': df[column].max(),
        'count': df[column].count(),
        'missing': df[column].isnull().sum()
    }
    
    return stats

def normalize_column(df, column, method='minmax'):
    """
    Normalize a column using specified method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to normalize
    method (str): Normalization method ('minmax' or 'zscore')
    
    Returns:
    pd.Series: Normalized column values
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    if method == 'minmax':
        col_min = df[column].min()
        col_max = df[column].max()
        if col_max == col_min:
            return pd.Series([0.5] * len(df), index=df.index)
        normalized = (df[column] - col_min) / (col_max - col_min)
    
    elif method == 'zscore':
        col_mean = df[column].mean()
        col_std = df[column].std()
        if col_std == 0:
            return pd.Series([0] * len(df), index=df.index)
        normalized = (df[column] - col_mean) / col_std
    
    else:
        raise ValueError("Method must be 'minmax' or 'zscore'")
    
    return normalized

def main():
    """
    Example usage of data cleaning functions.
    """
    np.random.seed(42)
    
    data = {
        'id': range(1, 101),
        'value': np.random.normal(100, 15, 100)
    }
    
    df = pd.DataFrame(data)
    
    print("Original DataFrame shape:", df.shape)
    print("\nOriginal summary statistics:")
    stats = calculate_summary_statistics(df, 'value')
    for key, value in stats.items():
        print(f"{key}: {value:.2f}")
    
    cleaned_df = remove_outliers_iqr(df, 'value')
    print("\nCleaned DataFrame shape:", cleaned_df.shape)
    
    print("\nCleaned summary statistics:")
    cleaned_stats = calculate_summary_statistics(cleaned_df, 'value')
    for key, value in cleaned_stats.items():
        print(f"{key}: {value:.2f}")
    
    normalized_values = normalize_column(cleaned_df, 'value', method='zscore')
    print(f"\nFirst 5 normalized values (z-score):")
    print(normalized_values.head())

if __name__ == "__main__":
    main()