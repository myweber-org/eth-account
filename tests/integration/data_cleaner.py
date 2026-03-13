import numpy as np
import pandas as pd

def remove_outliers_iqr(data, column, factor=1.5):
    """
    Remove outliers from a column using the IQR method.
    
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
    
    return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]

def normalize_minmax(data, column):
    """
    Normalize column values to [0, 1] range using min-max scaling.
    
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
    
    return (data[column] - min_val) / (max_val - min_val)

def standardize_zscore(data, column):
    """
    Standardize column values using z-score normalization.
    
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
    
    return (data[column] - mean_val) / std_val

def clean_dataset(data, numeric_columns=None, outlier_factor=1.5):
    """
    Clean dataset by removing outliers and normalizing numeric columns.
    
    Args:
        data: pandas DataFrame
        numeric_columns: list of numeric column names (default: all numeric columns)
        outlier_factor: IQR factor for outlier detection
    
    Returns:
        Cleaned DataFrame
    """
    if numeric_columns is None:
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_data = data.copy()
    
    for col in numeric_columns:
        if col in cleaned_data.columns:
            # Remove outliers
            cleaned_data = remove_outliers_iqr(cleaned_data, col, outlier_factor)
            
            # Normalize the column
            cleaned_data[f"{col}_normalized"] = normalize_minmax(cleaned_data, col)
            cleaned_data[f"{col}_standardized"] = standardize_zscore(cleaned_data, col)
    
    return cleaned_data

def get_summary_statistics(data, numeric_columns=None):
    """
    Get summary statistics for numeric columns.
    
    Args:
        data: pandas DataFrame
        numeric_columns: list of numeric column names
    
    Returns:
        DataFrame with summary statistics
    """
    if numeric_columns is None:
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    
    summary = pd.DataFrame()
    
    for col in numeric_columns:
        if col in data.columns:
            col_stats = {
                'mean': data[col].mean(),
                'median': data[col].median(),
                'std': data[col].std(),
                'min': data[col].min(),
                'max': data[col].max(),
                'q1': data[col].quantile(0.25),
                'q3': data[col].quantile(0.75),
                'iqr': data[col].quantile(0.75) - data[col].quantile(0.25),
                'skewness': data[col].skew(),
                'kurtosis': data[col].kurtosis()
            }
            summary[col] = pd.Series(col_stats)
    
    return summary.T
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a specified column using the Interquartile Range method.
    
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

def calculate_basic_stats(df, column):
    """
    Calculate basic statistics for a column.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to analyze
    
    Returns:
    dict: Dictionary containing statistical measures
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

def clean_numeric_data(df, columns=None):
    """
    Clean numeric data by removing NaN values and converting to appropriate types.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    columns (list): List of columns to clean. If None, cleans all numeric columns.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    if columns is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        columns = list(numeric_cols)
    
    cleaned_df = df.copy()
    
    for col in columns:
        if col in cleaned_df.columns:
            cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='coerce')
            cleaned_df[col].fillna(cleaned_df[col].median(), inplace=True)
    
    return cleaned_df

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, 3, 4, 5, 100],
        'B': [10, 20, 30, 40, 50, 60],
        'C': [1.1, 2.2, 3.3, 4.4, 5.5, 6.6]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print()
    
    cleaned_df = remove_outliers_iqr(df, 'A')
    print("DataFrame after removing outliers from column 'A':")
    print(cleaned_df)
    print()
    
    stats = calculate_basic_stats(df, 'B')
    print("Statistics for column 'B':")
    for key, value in stats.items():
        print(f"{key}: {value:.2f}")