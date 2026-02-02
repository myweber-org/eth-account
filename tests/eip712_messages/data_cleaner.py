
import numpy as np
import pandas as pd

def remove_outliers_iqr(data, column, multiplier=1.5):
    """
    Remove outliers using the Interquartile Range method.
    
    Args:
        data: pandas DataFrame
        column: column name to process
        multiplier: IQR multiplier (default 1.5)
    
    Returns:
        Filtered DataFrame without outliers
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data.copy()

def normalize_minmax(data, column):
    """
    Normalize data to [0, 1] range using min-max scaling.
    
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
    
    normalized = (data[column] - min_val) / (max_val - min_val)
    return normalized

def standardize_zscore(data, column):
    """
    Standardize data using z-score normalization.
    
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
    
    standardized = (data[column] - mean_val) / std_val
    return standardized

def clean_missing_values(data, strategy='mean'):
    """
    Handle missing values in numeric columns.
    
    Args:
        data: pandas DataFrame
        strategy: imputation strategy ('mean', 'median', 'mode', or 'drop')
    
    Returns:
        DataFrame with handled missing values
    """
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    
    if strategy == 'drop':
        cleaned_data = data.dropna(subset=numeric_cols)
    elif strategy == 'mean':
        cleaned_data = data.copy()
        for col in numeric_cols:
            cleaned_data[col].fillna(data[col].mean(), inplace=True)
    elif strategy == 'median':
        cleaned_data = data.copy()
        for col in numeric_cols:
            cleaned_data[col].fillna(data[col].median(), inplace=True)
    elif strategy == 'mode':
        cleaned_data = data.copy()
        for col in numeric_cols:
            cleaned_data[col].fillna(data[col].mode()[0], inplace=True)
    else:
        raise ValueError("Strategy must be 'mean', 'median', 'mode', or 'drop'")
    
    return cleaned_data

def process_dataframe(df, numeric_columns=None, outlier_multiplier=1.5, 
                     normalization_method='standardize', missing_strategy='mean'):
    """
    Complete data cleaning pipeline for numeric columns.
    
    Args:
        df: Input DataFrame
        numeric_columns: list of numeric columns to process (default: all numeric)
        outlier_multiplier: multiplier for IQR outlier detection
        normalization_method: 'standardize', 'normalize', or None
        missing_strategy: strategy for handling missing values
    
    Returns:
        Cleaned and processed DataFrame
    """
    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    processed_df = df.copy()
    
    # Handle missing values
    processed_df = clean_missing_values(processed_df, strategy=missing_strategy)
    
    # Remove outliers for each numeric column
    for col in numeric_columns:
        if col in processed_df.columns:
            processed_df = remove_outliers_iqr(processed_df, col, outlier_multiplier)
    
    # Apply normalization
    for col in numeric_columns:
        if col in processed_df.columns:
            if normalization_method == 'standardize':
                processed_df[f'{col}_standardized'] = standardize_zscore(processed_df, col)
            elif normalization_method == 'normalize':
                processed_df[f'{col}_normalized'] = normalize_minmax(processed_df, col)
    
    return processed_dfimport pandas as pd
import numpy as np

def clean_dataset(df, strategy='mean', outlier_threshold=3):
    """
    Clean a pandas DataFrame by handling missing values and outliers.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean
        strategy (str): Strategy for missing values ('mean', 'median', 'mode', 'drop')
        outlier_threshold (float): Z-score threshold for outlier detection
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    df_clean = df.copy()
    
    # Handle missing values
    if strategy == 'mean':
        df_clean = df_clean.fillna(df_clean.mean())
    elif strategy == 'median':
        df_clean = df_clean.fillna(df_clean.median())
    elif strategy == 'mode':
        df_clean = df_clean.fillna(df_clean.mode().iloc[0])
    elif strategy == 'drop':
        df_clean = df_clean.dropna()
    
    # Handle outliers using Z-score method
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        z_scores = np.abs((df_clean[col] - df_clean[col].mean()) / df_clean[col].std())
        df_clean = df_clean[z_scores < outlier_threshold]
    
    # Reset index after outlier removal
    df_clean = df_clean.reset_index(drop=True)
    
    return df_clean

def remove_duplicates(df, subset=None, keep='first'):
    """
    Remove duplicate rows from DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        subset (list): Columns to consider for duplicates
        keep (str): Which duplicates to keep ('first', 'last', False)
    
    Returns:
        pd.DataFrame: DataFrame without duplicates
    """
    return df.drop_duplicates(subset=subset, keep=keep)

def normalize_data(df, method='minmax'):
    """
    Normalize numeric columns in DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        method (str): Normalization method ('minmax', 'zscore')
    
    Returns:
        pd.DataFrame: Normalized DataFrame
    """
    df_norm = df.copy()
    numeric_cols = df_norm.select_dtypes(include=[np.number]).columns
    
    if method == 'minmax':
        for col in numeric_cols:
            df_norm[col] = (df_norm[col] - df_norm[col].min()) / (df_norm[col].max() - df_norm[col].min())
    elif method == 'zscore':
        for col in numeric_cols:
            df_norm[col] = (df_norm[col] - df_norm[col].mean()) / df_norm[col].std()
    
    return df_norm

# Example usage
if __name__ == "__main__":
    # Create sample data
    data = {
        'A': [1, 2, np.nan, 4, 5, 100],
        'B': [10, 20, 30, np.nan, 50, 60],
        'C': ['x', 'y', 'x', 'y', 'x', 'y']
    }
    
    df = pd.DataFrame(data)
    print("Original DataFrame:")
    print(df)
    
    cleaned_df = clean_dataset(df, strategy='mean', outlier_threshold=2)
    print("\nCleaned DataFrame:")
    print(cleaned_df)
    
    normalized_df = normalize_data(cleaned_df, method='minmax')
    print("\nNormalized DataFrame:")
    print(normalized_df)