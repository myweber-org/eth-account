
import pandas as pd
import numpy as np

def clean_missing_values(df, strategy='mean', columns=None):
    """
    Clean missing values in a DataFrame using specified strategy.
    
    Args:
        df: pandas DataFrame
        strategy: Method for imputation ('mean', 'median', 'mode', 'drop')
        columns: List of columns to clean, if None cleans all columns
    
    Returns:
        Cleaned DataFrame
    """
    if df.empty:
        return df
    
    if columns is None:
        columns = df.columns
    
    df_clean = df.copy()
    
    for col in columns:
        if col not in df_clean.columns:
            continue
            
        if df_clean[col].isnull().any():
            if strategy == 'drop':
                df_clean = df_clean.dropna(subset=[col])
            elif strategy == 'mean' and pd.api.types.is_numeric_dtype(df_clean[col]):
                df_clean[col] = df_clean[col].fillna(df_clean[col].mean())
            elif strategy == 'median' and pd.api.types.is_numeric_dtype(df_clean[col]):
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())
            elif strategy == 'mode':
                df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])
            else:
                df_clean[col] = df_clean[col].fillna(method='ffill').fillna(method='bfill')
    
    return df_clean

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Args:
        df: pandas DataFrame to validate
        required_columns: List of required column names
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not isinstance(df, pd.DataFrame):
        return False, "Input is not a pandas DataFrame"
    
    if df.empty:
        return False, "DataFrame is empty"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    return True, "DataFrame is valid"

def load_and_clean_csv(filepath, **kwargs):
    """
    Load CSV file and clean missing values.
    
    Args:
        filepath: Path to CSV file
        **kwargs: Additional arguments for clean_missing_values
    
    Returns:
        Cleaned DataFrame
    """
    try:
        df = pd.read_csv(filepath)
        is_valid, message = validate_dataframe(df)
        
        if not is_valid:
            raise ValueError(f"Invalid DataFrame: {message}")
        
        return clean_missing_values(df, **kwargs)
    except Exception as e:
        print(f"Error loading or cleaning file: {e}")
        return pd.DataFrame()

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'A': [1, 2, np.nan, 4, 5],
        'B': [np.nan, 2, 3, np.nan, 5],
        'C': [1, 2, 3, 4, 5]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nCleaned with mean imputation:")
    print(clean_missing_values(df, strategy='mean'))
    print("\nCleaned with drop strategy:")
    print(clean_missing_values(df, strategy='drop'))