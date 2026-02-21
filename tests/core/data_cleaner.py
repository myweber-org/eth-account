import pandas as pd
import numpy as np

def clean_dataset(df, numeric_columns=None, method='median', outlier_threshold=3):
    """
    Clean dataset by handling missing values and removing outliers.
    
    Parameters:
    df (pd.DataFrame): Input dataframe
    numeric_columns (list): List of numeric column names to process
    method (str): Imputation method ('mean', 'median', 'mode')
    outlier_threshold (float): Z-score threshold for outlier detection
    
    Returns:
    pd.DataFrame: Cleaned dataframe
    """
    if df.empty:
        return df
    
    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    df_clean = df.copy()
    
    # Handle missing values
    for col in numeric_columns:
        if col in df_clean.columns:
            if method == 'mean':
                fill_value = df_clean[col].mean()
            elif method == 'median':
                fill_value = df_clean[col].median()
            elif method == 'mode':
                fill_value = df_clean[col].mode()[0] if not df_clean[col].mode().empty else 0
            else:
                fill_value = 0
            
            df_clean[col].fillna(fill_value, inplace=True)
    
    # Remove outliers using Z-score method
    if outlier_threshold > 0:
        z_scores = np.abs((df_clean[numeric_columns] - df_clean[numeric_columns].mean()) / 
                         df_clean[numeric_columns].std())
        outlier_mask = (z_scores < outlier_threshold).all(axis=1)
        df_clean = df_clean[outlier_mask].reset_index(drop=True)
    
    return df_clean

def validate_data(df, required_columns=None, min_rows=1):
    """
    Validate dataframe structure and content.
    
    Parameters:
    df (pd.DataFrame): Dataframe to validate
    required_columns (list): List of required column names
    min_rows (int): Minimum number of rows required
    
    Returns:
    tuple: (is_valid, error_message)
    """
    if not isinstance(df, pd.DataFrame):
        return False, "Input is not a pandas DataFrame"
    
    if len(df) < min_rows:
        return False, f"Dataframe has fewer than {min_rows} rows"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    return True, "Data validation passed"

def normalize_columns(df, columns=None, method='minmax'):
    """
    Normalize specified columns in dataframe.
    
    Parameters:
    df (pd.DataFrame): Input dataframe
    columns (list): Columns to normalize
    method (str): Normalization method ('minmax' or 'zscore')
    
    Returns:
    pd.DataFrame: Dataframe with normalized columns
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    df_normalized = df.copy()
    
    for col in columns:
        if col in df_normalized.columns and pd.api.types.is_numeric_dtype(df_normalized[col]):
            if method == 'minmax':
                col_min = df_normalized[col].min()
                col_max = df_normalized[col].max()
                if col_max > col_min:
                    df_normalized[col] = (df_normalized[col] - col_min) / (col_max - col_min)
            elif method == 'zscore':
                col_mean = df_normalized[col].mean()
                col_std = df_normalized[col].std()
                if col_std > 0:
                    df_normalized[col] = (df_normalized[col] - col_mean) / col_std
    
    return df_normalized

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'A': [1, 2, np.nan, 4, 100],
        'B': [5, 6, 7, np.nan, 9],
        'C': ['x', 'y', 'z', 'x', 'y']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    
    cleaned_df = clean_dataset(df, numeric_columns=['A', 'B'])
    print("\nCleaned DataFrame:")
    print(cleaned_df)
    
    is_valid, message = validate_data(cleaned_df, required_columns=['A', 'B'])
    print(f"\nValidation: {is_valid} - {message}")
    
    normalized_df = normalize_columns(cleaned_df, method='minmax')
    print("\nNormalized DataFrame:")
    print(normalized_df)