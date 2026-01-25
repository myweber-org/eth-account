import pandas as pd
import numpy as np

def clean_missing_data(file_path, strategy='mean', columns=None):
    """
    Load a CSV file and handle missing values using specified strategy.
    
    Args:
        file_path (str): Path to the CSV file.
        strategy (str): Method for handling missing values. 
                       Options: 'mean', 'median', 'mode', 'drop'.
        columns (list): Specific columns to clean. If None, clean all columns.
    
    Returns:
        pandas.DataFrame: Cleaned dataframe.
    """
    
    df = pd.read_csv(file_path)
    
    if columns is None:
        columns = df.columns
    
    for column in columns:
        if column in df.columns:
            if strategy == 'mean':
                df[column].fillna(df[column].mean(), inplace=True)
            elif strategy == 'median':
                df[column].fillna(df[column].median(), inplace=True)
            elif strategy == 'mode':
                df[column].fillna(df[column].mode()[0], inplace=True)
            elif strategy == 'drop':
                df.dropna(subset=[column], inplace=True)
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
    
    return df

def detect_outliers_iqr(df, column, threshold=1.5):
    """
    Detect outliers using Interquartile Range method.
    
    Args:
        df (pandas.DataFrame): Input dataframe.
        column (str): Column name to check for outliers.
        threshold (float): IQR multiplier threshold.
    
    Returns:
        pandas.Series: Boolean series indicating outliers.
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR
    
    outliers = (df[column] < lower_bound) | (df[column] > upper_bound)
    return outliers

def normalize_column(df, column, method='minmax'):
    """
    Normalize a column using specified method.
    
    Args:
        df (pandas.DataFrame): Input dataframe.
        column (str): Column name to normalize.
        method (str): Normalization method. Options: 'minmax', 'zscore'.
    
    Returns:
        pandas.DataFrame: Dataframe with normalized column.
    """
    df_copy = df.copy()
    
    if method == 'minmax':
        min_val = df_copy[column].min()
        max_val = df_copy[column].max()
        df_copy[column] = (df_copy[column] - min_val) / (max_val - min_val)
    elif method == 'zscore':
        mean_val = df_copy[column].mean()
        std_val = df_copy[column].std()
        df_copy[column] = (df_copy[column] - mean_val) / std_val
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return df_copy

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, np.nan, 4, 5],
        'B': [10, np.nan, 30, 40, 50],
        'C': [100, 200, 300, 400, 500]
    }
    
    df_sample = pd.DataFrame(sample_data)
    df_sample.to_csv('sample_data.csv', index=False)
    
    cleaned_df = clean_missing_data('sample_data.csv', strategy='mean')
    print("Cleaned DataFrame:")
    print(cleaned_df)
    
    outliers = detect_outliers_iqr(cleaned_df, 'A')
    print(f"\nOutliers in column A: {outliers.sum()}")
    
    normalized_df = normalize_column(cleaned_df, 'A', method='minmax')
    print("\nNormalized column A:")
    print(normalized_df['A'])