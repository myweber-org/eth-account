
import pandas as pd

def remove_duplicates(df, subset=None, keep='first'):
    """
    Remove duplicate rows from a DataFrame.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    subset (list, optional): Column names to consider for duplicates
    keep (str): Which duplicates to keep - 'first', 'last', or False
    
    Returns:
    pd.DataFrame: DataFrame with duplicates removed
    """
    if df.empty:
        return df
    
    cleaned_df = df.drop_duplicates(subset=subset, keep=keep)
    removed_count = len(df) - len(cleaned_df)
    
    if removed_count > 0:
        print(f"Removed {removed_count} duplicate rows")
    
    return cleaned_df.reset_index(drop=True)

def clean_numeric_columns(df, columns):
    """
    Clean numeric columns by converting to appropriate types.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    columns (list): List of column names to clean
    
    Returns:
    pd.DataFrame: DataFrame with cleaned numeric columns
    """
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    required_columns (list, optional): List of required column names
    
    Returns:
    bool: True if validation passes, False otherwise
    """
    if df.empty:
        print("Warning: DataFrame is empty")
        return False
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Missing required columns: {missing_columns}")
            return False
    
    return True

def get_data_summary(df):
    """
    Generate summary statistics for the DataFrame.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    
    Returns:
    dict: Dictionary containing summary statistics
    """
    summary = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': df.isnull().sum().sum(),
        'duplicate_rows': df.duplicated().sum(),
        'data_types': df.dtypes.to_dict()
    }
    
    return summaryimport pandas as pd
import numpy as np

def remove_duplicates(df, subset=None, keep='first'):
    """
    Remove duplicate rows from a DataFrame.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    subset (list, optional): Columns to consider for duplicates
    keep (str): Which duplicates to keep - 'first', 'last', or False
    
    Returns:
    pd.DataFrame: DataFrame with duplicates removed
    """
    if df.empty:
        return df
    
    if subset is None:
        subset = df.columns.tolist()
    
    cleaned_df = df.drop_duplicates(subset=subset, keep=keep)
    
    print(f"Removed {len(df) - len(cleaned_df)} duplicate rows")
    print(f"Original shape: {df.shape}, Cleaned shape: {cleaned_df.shape}")
    
    return cleaned_df

def clean_missing_values(df, strategy='drop', fill_value=None):
    """
    Handle missing values in DataFrame.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    strategy (str): 'drop' to remove rows, 'fill' to fill values
    fill_value: Value to fill missing values with
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    if df.empty:
        return df
    
    if strategy == 'drop':
        cleaned_df = df.dropna()
        print(f"Removed {len(df) - len(cleaned_df)} rows with missing values")
    elif strategy == 'fill':
        if fill_value is None:
            fill_value = df.mean(numeric_only=True)
        cleaned_df = df.fillna(fill_value)
        print("Filled missing values")
    else:
        raise ValueError("Strategy must be 'drop' or 'fill'")
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate
    required_columns (list): List of required column names
    
    Returns:
    bool: True if validation passes
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    
    if df.empty:
        print("Warning: DataFrame is empty")
        return True
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
    
    print(f"DataFrame validation passed")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    return True

def process_dataframe(df, 
                     remove_dups=True, 
                     handle_missing=True, 
                     missing_strategy='drop',
                     required_columns=None):
    """
    Main function to process and clean DataFrame.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    remove_dups (bool): Whether to remove duplicates
    handle_missing (bool): Whether to handle missing values
    missing_strategy (str): Strategy for missing values
    required_columns (list): Required columns for validation
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    print("Starting data cleaning process...")
    
    validate_dataframe(df, required_columns)
    
    if remove_dups:
        df = remove_duplicates(df)
    
    if handle_missing:
        df = clean_missing_values(df, strategy=missing_strategy)
    
    print("Data cleaning completed successfully")
    return df

if __name__ == "__main__":
    sample_data = {
        'id': [1, 2, 2, 3, 4, 4, 5],
        'name': ['Alice', 'Bob', 'Bob', 'Charlie', None, 'Eve', 'Frank'],
        'score': [85, 90, 90, 78, 92, 92, 88],
        'age': [25, 30, 30, 22, None, 28, 35]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    cleaned_df = process_dataframe(
        df,
        remove_dups=True,
        handle_missing=True,
        missing_strategy='drop',
        required_columns=['id', 'name', 'score']
    )
    
    print("\nCleaned DataFrame:")
    print(cleaned_df)