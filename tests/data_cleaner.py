
import pandas as pd
import numpy as np

def clean_dataset(df, strategy='mean', outlier_threshold=3):
    """
    Clean a pandas DataFrame by handling missing values and outliers.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    strategy (str): Strategy for missing value imputation ('mean', 'median', 'mode').
    outlier_threshold (float): Number of standard deviations for outlier detection.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    # Handle missing values
    numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        if cleaned_df[col].isnull().any():
            if strategy == 'mean':
                fill_value = cleaned_df[col].mean()
            elif strategy == 'median':
                fill_value = cleaned_df[col].median()
            elif strategy == 'mode':
                fill_value = cleaned_df[col].mode()[0]
            else:
                fill_value = 0
            cleaned_df[col].fillna(fill_value, inplace=True)
    
    # Handle outliers using z-score method
    for col in numeric_cols:
        z_scores = np.abs((cleaned_df[col] - cleaned_df[col].mean()) / cleaned_df[col].std())
        outliers = z_scores > outlier_threshold
        if outliers.any():
            median_value = cleaned_df[col].median()
            cleaned_df.loc[outliers, col] = median_value
    
    return cleaned_df

def remove_duplicates(df, subset=None, keep='first'):
    """
    Remove duplicate rows from DataFrame.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame.
    subset (list): Columns to consider for duplicate detection.
    keep (str): Which duplicates to keep ('first', 'last', False).
    
    Returns:
    pd.DataFrame: DataFrame with duplicates removed.
    """
    return df.drop_duplicates(subset=subset, keep=keep)

def normalize_columns(df, columns=None):
    """
    Normalize specified columns to range [0, 1].
    
    Parameters:
    df (pd.DataFrame): Input DataFrame.
    columns (list): Columns to normalize. If None, normalize all numeric columns.
    
    Returns:
    pd.DataFrame: DataFrame with normalized columns.
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    normalized_df = df.copy()
    
    for col in columns:
        if col in normalized_df.columns:
            col_min = normalized_df[col].min()
            col_max = normalized_df[col].max()
            if col_max != col_min:
                normalized_df[col] = (normalized_df[col] - col_min) / (col_max - col_min)
    
    return normalized_df

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'A': [1, 2, np.nan, 4, 100],
        'B': [5, 6, 7, 8, 9],
        'C': [10, 11, 12, 13, 14]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    
    cleaned = clean_dataset(df, strategy='median', outlier_threshold=2)
    print("\nCleaned DataFrame:")
    print(cleaned)
    
    normalized = normalize_columns(cleaned, columns=['A', 'B'])
    print("\nNormalized DataFrame:")
    print(normalized)