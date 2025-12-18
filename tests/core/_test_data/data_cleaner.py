import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range method.
    
    Args:
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
    
    return filtered_df

def calculate_basic_stats(df, column):
    """
    Calculate basic statistics for a DataFrame column.
    
    Args:
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
    Clean numeric data by removing outliers from specified columns.
    If no columns specified, clean all numeric columns.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        columns (list, optional): List of column names to clean
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_df = df.copy()
    
    for column in columns:
        if column in cleaned_df.columns and pd.api.types.is_numeric_dtype(cleaned_df[column]):
            try:
                cleaned_df = remove_outliers_iqr(cleaned_df, column)
            except Exception as e:
                print(f"Warning: Could not clean column '{column}': {e}")
    
    return cleaned_df

def generate_cleaning_report(original_df, cleaned_df):
    """
    Generate a report comparing original and cleaned DataFrames.
    
    Args:
        original_df (pd.DataFrame): Original DataFrame
        cleaned_df (pd.DataFrame): Cleaned DataFrame
    
    Returns:
        pd.DataFrame: Report DataFrame
    """
    numeric_cols = original_df.select_dtypes(include=[np.number]).columns
    
    report_data = []
    
    for col in numeric_cols:
        if col in cleaned_df.columns:
            original_stats = calculate_basic_stats(original_df, col)
            cleaned_stats = calculate_basic_stats(cleaned_df, col)
            
            report_data.append({
                'column': col,
                'original_rows': original_stats['count'],
                'cleaned_rows': cleaned_stats['count'],
                'rows_removed': original_stats['count'] - cleaned_stats['count'],
                'original_mean': original_stats['mean'],
                'cleaned_mean': cleaned_stats['mean'],
                'original_std': original_stats['std'],
                'cleaned_std': cleaned_stats['std']
            })
    
    report_df = pd.DataFrame(report_data)
    
    return report_df

if __name__ == "__main__":
    sample_data = {
        'A': np.random.normal(100, 15, 1000),
        'B': np.random.exponential(50, 1000),
        'C': np.random.uniform(0, 200, 1000)
    }
    
    df = pd.DataFrame(sample_data)
    df.loc[np.random.choice(df.index, 50), 'A'] = np.random.uniform(500, 1000, 50)
    
    print("Original DataFrame shape:", df.shape)
    print("\nOriginal statistics:")
    for col in df.columns:
        stats = calculate_basic_stats(df, col)
        print(f"{col}: mean={stats['mean']:.2f}, std={stats['std']:.2f}")
    
    cleaned_df = clean_numeric_data(df)
    
    print("\nCleaned DataFrame shape:", cleaned_df.shape)
    print("\nCleaned statistics:")
    for col in cleaned_df.columns:
        stats = calculate_basic_stats(cleaned_df, col)
        print(f"{col}: mean={stats['mean']:.2f}, std={stats['std']:.2f}")
    
    report = generate_cleaning_report(df, cleaned_df)
    print("\nCleaning Report:")
    print(report)
import numpy as np
import pandas as pd

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
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data

def normalize_minmax(data, column):
    """
    Normalize column values to range [0, 1]
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    min_val = data[column].min()
    max_val = data[column].max()
    
    if min_val == max_val:
        return data[column].apply(lambda x: 0.5)
    
    normalized = (data[column] - min_val) / (max_val - min_val)
    return normalized

def standardize_zscore(data, column):
    """
    Standardize column values using z-score
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    mean_val = data[column].mean()
    std_val = data[column].std()
    
    if std_val == 0:
        return data[column].apply(lambda x: 0)
    
    standardized = (data[column] - mean_val) / std_val
    return standardized

def handle_missing_values(data, strategy='mean', columns=None):
    """
    Handle missing values in specified columns
    """
    if columns is None:
        columns = data.columns
    
    data_filled = data.copy()
    
    for column in columns:
        if column not in data.columns:
            continue
            
        if data[column].isnull().any():
            if strategy == 'mean':
                fill_value = data[column].mean()
            elif strategy == 'median':
                fill_value = data[column].median()
            elif strategy == 'mode':
                fill_value = data[column].mode()[0]
            elif strategy == 'drop':
                data_filled = data_filled.dropna(subset=[column])
                continue
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
            
            data_filled[column] = data_filled[column].fillna(fill_value)
    
    return data_filled

def clean_dataset(data, config):
    """
    Apply multiple cleaning operations based on configuration
    """
    cleaned_data = data.copy()
    
    for column, operations in config.items():
        if column not in cleaned_data.columns:
            continue
            
        for operation in operations:
            if operation['type'] == 'remove_outliers':
                cleaned_data = remove_outliers_iqr(
                    cleaned_data, 
                    column, 
                    multiplier=operation.get('multiplier', 1.5)
                )
            elif operation['type'] == 'normalize':
                cleaned_data[column] = normalize_minmax(cleaned_data, column)
            elif operation['type'] == 'standardize':
                cleaned_data[column] = standardize_zscore(cleaned_data, column)
            elif operation['type'] == 'fill_missing':
                cleaned_data = handle_missing_values(
                    cleaned_data,
                    strategy=operation.get('strategy', 'mean'),
                    columns=[column]
                )
    
    return cleaned_data