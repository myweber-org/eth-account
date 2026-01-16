import numpy as np
import pandas as pd

def remove_outliers_iqr(data, column, factor=1.5):
    """
    Remove outliers from a DataFrame column using the IQR method.
    
    Args:
        data: pandas DataFrame
        column: Column name to process
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
    return filtered_data.copy()

def normalize_minmax(data, column):
    """
    Normalize a column using min-max scaling to [0, 1] range.
    
    Args:
        data: pandas DataFrame
        column: Column name to normalize
    
    Returns:
        Series with normalized values
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    col_min = data[column].min()
    col_max = data[column].max()
    
    if col_max == col_min:
        return pd.Series([0.5] * len(data), index=data.index)
    
    normalized = (data[column] - col_min) / (col_max - col_min)
    return normalized

def clean_dataset(df, numeric_columns=None, outlier_factor=1.5):
    """
    Clean dataset by removing outliers and normalizing numeric columns.
    
    Args:
        df: Input DataFrame
        numeric_columns: List of numeric columns to process (default: all numeric)
        outlier_factor: IQR factor for outlier detection
    
    Returns:
        Cleaned DataFrame
    """
    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_df = df.copy()
    
    for col in numeric_columns:
        if col in df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, col, outlier_factor)
            cleaned_df[f"{col}_normalized"] = normalize_minmax(cleaned_df, col)
    
    return cleaned_df.reset_index(drop=True)

def calculate_statistics(df, column):
    """
    Calculate basic statistics for a column.
    
    Args:
        df: pandas DataFrame
        column: Column name
    
    Returns:
        Dictionary with statistics
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

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'id': range(1, 101),
        'value': np.random.normal(100, 15, 100),
        'category': np.random.choice(['A', 'B', 'C'], 100)
    })
    
    print("Original data shape:", sample_data.shape)
    print("Original statistics:", calculate_statistics(sample_data, 'value'))
    
    cleaned = clean_dataset(sample_data, ['value'])
    print("\nCleaned data shape:", cleaned.shape)
    print("Cleaned statistics:", calculate_statistics(cleaned, 'value'))