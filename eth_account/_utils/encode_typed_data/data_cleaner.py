import pandas as pd
import numpy as np

def clean_csv_data(filepath, fill_strategy='mean'):
    """
    Load a CSV file and clean missing values.
    
    Args:
        filepath (str): Path to the CSV file.
        fill_strategy (str): Strategy for filling missing values.
            Options: 'mean', 'median', 'mode', 'zero', 'drop'.
    
    Returns:
        pandas.DataFrame: Cleaned DataFrame.
    """
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {filepath}")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if fill_strategy == 'mean':
        for col in numeric_cols:
            df[col] = df[col].fillna(df[col].mean())
    elif fill_strategy == 'median':
        for col in numeric_cols:
            df[col] = df[col].fillna(df[col].median())
    elif fill_strategy == 'mode':
        for col in numeric_cols:
            df[col] = df[col].fillna(df[col].mode()[0])
    elif fill_strategy == 'zero':
        df[numeric_cols] = df[numeric_cols].fillna(0)
    elif fill_strategy == 'drop':
        df = df.dropna(subset=numeric_cols)
    else:
        raise ValueError(f"Unknown fill strategy: {fill_strategy}")
    
    return df

def remove_outliers_iqr(df, column, multiplier=1.5):
    """
    Remove outliers from a column using the IQR method.
    
    Args:
        df (pandas.DataFrame): Input DataFrame.
        column (str): Column name to process.
        multiplier (float): IQR multiplier for outlier detection.
    
    Returns:
        pandas.DataFrame: DataFrame with outliers removed.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def standardize_columns(df, columns=None):
    """
    Standardize specified columns to have zero mean and unit variance.
    
    Args:
        df (pandas.DataFrame): Input DataFrame.
        columns (list): List of column names to standardize.
            If None, standardize all numeric columns.
    
    Returns:
        pandas.DataFrame: DataFrame with standardized columns.
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    for col in columns:
        if col in df.columns and np.issubdtype(df[col].dtype, np.number):
            mean = df[col].mean()
            std = df[col].std()
            if std > 0:
                df[col] = (df[col] - mean) / std
    
    return df

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, np.nan, 4, 5],
        'B': [10, np.nan, 30, 40, 50],
        'C': ['x', 'y', 'z', 'x', 'y']
    }
    
    test_df = pd.DataFrame(sample_data)
    test_df.to_csv('test_data.csv', index=False)
    
    cleaned = clean_csv_data('test_data.csv', fill_strategy='mean')
    print("Cleaned DataFrame:")
    print(cleaned)
    
    import os
    os.remove('test_data.csv')