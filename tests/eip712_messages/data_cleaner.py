import pandas as pd
import numpy as np

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

def calculate_summary_statistics(df):
    """
    Calculate summary statistics for numerical columns.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    
    Returns:
    pd.DataFrame: Summary statistics
    """
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    
    if len(numerical_cols) == 0:
        return pd.DataFrame()
    
    stats = df[numerical_cols].agg(['count', 'mean', 'std', 'min', 'max'])
    
    return stats.T

def clean_missing_values(df, strategy='mean'):
    """
    Handle missing values in numerical columns.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    strategy (str): Imputation strategy ('mean', 'median', 'mode', or 'drop')
    
    Returns:
    pd.DataFrame: DataFrame with handled missing values
    """
    df_clean = df.copy()
    numerical_cols = df_clean.select_dtypes(include=[np.number]).columns
    
    for col in numerical_cols:
        if df_clean[col].isnull().any():
            if strategy == 'mean':
                fill_value = df_clean[col].mean()
            elif strategy == 'median':
                fill_value = df_clean[col].median()
            elif strategy == 'mode':
                fill_value = df_clean[col].mode()[0]
            elif strategy == 'drop':
                df_clean = df_clean.dropna(subset=[col])
                continue
            else:
                raise ValueError("Strategy must be 'mean', 'median', 'mode', or 'drop'")
            
            df_clean[col] = df_clean[col].fillna(fill_value)
    
    return df_clean

def normalize_data(df, columns=None):
    """
    Normalize specified columns using min-max scaling.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    columns (list): List of columns to normalize. If None, normalize all numerical columns.
    
    Returns:
    pd.DataFrame: DataFrame with normalized columns
    """
    df_normalized = df.copy()
    
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    for col in columns:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            min_val = df_normalized[col].min()
            max_val = df_normalized[col].max()
            
            if max_val > min_val:
                df_normalized[col] = (df_normalized[col] - min_val) / (max_val - min_val)
    
    return df_normalized

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
    
    stats = calculate_summary_statistics(df)
    print("Summary Statistics:")
    print(stats)
    print()
    
    df_with_nulls = df.copy()
    df_with_nulls.loc[2, 'B'] = np.nan
    df_cleaned = clean_missing_values(df_with_nulls, strategy='mean')
    print("DataFrame after handling missing values:")
    print(df_cleaned)
    print()
    
    normalized_df = normalize_data(df, columns=['A', 'B'])
    print("Normalized DataFrame:")
    print(normalized_df)