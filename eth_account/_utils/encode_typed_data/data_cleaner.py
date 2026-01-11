import pandas as pd

def remove_duplicates(df, subset=None, keep='first'):
    """
    Remove duplicate rows from a DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        subset (list, optional): Column labels to consider for duplicates.
        keep (str, optional): Which duplicates to keep.
    
    Returns:
        pd.DataFrame: DataFrame with duplicates removed.
    """
    if df.empty:
        return df
    
    cleaned_df = df.drop_duplicates(subset=subset, keep=keep)
    
    removed_count = len(df) - len(cleaned_df)
    if removed_count > 0:
        print(f"Removed {removed_count} duplicate rows.")
    
    return cleaned_df.reset_index(drop=True)

def validate_dataframe(df):
    """
    Perform basic validation on DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame to validate.
    
    Returns:
        bool: True if DataFrame passes validation.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")
    
    if df.empty:
        print("Warning: DataFrame is empty.")
        return False
    
    return True

def clean_dataset(file_path, output_path=None):
    """
    Load, clean, and save a dataset.
    
    Args:
        file_path (str): Path to input CSV file.
        output_path (str, optional): Path for cleaned output.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    
    if not validate_dataframe(df):
        return None
    
    # Remove duplicates
    df_clean = remove_duplicates(df)
    
    # Remove rows with all NaN values
    df_clean = df_clean.dropna(how='all')
    
    # Reset index
    df_clean = df_clean.reset_index(drop=True)
    
    if output_path:
        df_clean.to_csv(output_path, index=False)
        print(f"Cleaned data saved to {output_path}")
    
    return df_cleanimport pandas as pd
import numpy as np

def remove_duplicates(df, subset=None):
    """
    Remove duplicate rows from DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        subset (list, optional): Columns to consider for duplicates
    
    Returns:
        pd.DataFrame: DataFrame with duplicates removed
    """
    return df.drop_duplicates(subset=subset, keep='first')

def fill_missing_values(df, strategy='mean', columns=None):
    """
    Fill missing values in DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        strategy (str): 'mean', 'median', 'mode', or 'constant'
        columns (list): Specific columns to fill
    
    Returns:
        pd.DataFrame: DataFrame with filled values
    """
    df_filled = df.copy()
    
    if columns is None:
        columns = df.columns
    
    for col in columns:
        if df[col].dtype in ['int64', 'float64']:
            if strategy == 'mean':
                fill_value = df[col].mean()
            elif strategy == 'median':
                fill_value = df[col].median()
            elif strategy == 'mode':
                fill_value = df[col].mode()[0] if not df[col].mode().empty else 0
            elif strategy == 'constant':
                fill_value = 0
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
            
            df_filled[col] = df[col].fillna(fill_value)
    
    return df_filled

def normalize_columns(df, columns=None, method='minmax'):
    """
    Normalize specified columns in DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        columns (list): Columns to normalize
        method (str): 'minmax' or 'zscore'
    
    Returns:
        pd.DataFrame: DataFrame with normalized columns
    """
    df_normalized = df.copy()
    
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    for col in columns:
        if method == 'minmax':
            min_val = df[col].min()
            max_val = df[col].max()
            if max_val > min_val:
                df_normalized[col] = (df[col] - min_val) / (max_val - min_val)
        
        elif method == 'zscore':
            mean_val = df[col].mean()
            std_val = df[col].std()
            if std_val > 0:
                df_normalized[col] = (df[col] - mean_val) / std_val
    
    return df_normalized

def detect_outliers(df, columns=None, threshold=3):
    """
    Detect outliers using z-score method.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        columns (list): Columns to check for outliers
        threshold (float): Z-score threshold
    
    Returns:
        pd.DataFrame: Boolean mask of outliers
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    outliers = pd.DataFrame(False, index=df.index, columns=columns)
    
    for col in columns:
        z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
        outliers[col] = z_scores > threshold
    
    return outliers

def clean_dataframe(df, operations=None):
    """
    Apply multiple cleaning operations to DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        operations (list): List of cleaning operations to apply
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    if operations is None:
        operations = [
            ('remove_duplicates', {}),
            ('fill_missing_values', {'strategy': 'mean'}),
            ('normalize_columns', {'method': 'minmax'})
        ]
    
    cleaned_df = df.copy()
    
    for operation, params in operations:
        if operation == 'remove_duplicates':
            cleaned_df = remove_duplicates(cleaned_df, **params)
        elif operation == 'fill_missing_values':
            cleaned_df = fill_missing_values(cleaned_df, **params)
        elif operation == 'normalize_columns':
            cleaned_df = normalize_columns(cleaned_df, **params)
        elif operation == 'detect_outliers':
            outlier_mask = detect_outliers(cleaned_df, **params)
            cleaned_df = cleaned_df[~outlier_mask.any(axis=1)]
    
    return cleaned_df