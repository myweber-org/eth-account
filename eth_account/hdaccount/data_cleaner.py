import pandas as pd

def clean_dataset(df, column_names):
    """
    Clean a pandas DataFrame by removing duplicates and normalizing specified string columns.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        column_names (list): List of column names to normalize (strip whitespace, lower case).
    
    Returns:
        pd.DataFrame: Cleaned DataFrame with duplicates removed and strings normalized.
    """
    cleaned_df = df.copy()
    
    # Remove duplicate rows
    cleaned_df = cleaned_df.drop_duplicates()
    
    # Normalize string columns
    for col in column_names:
        if col in cleaned_df.columns and cleaned_df[col].dtype == 'object':
            cleaned_df[col] = cleaned_df[col].str.strip().str.lower()
    
    # Reset index after cleaning
    cleaned_df = cleaned_df.reset_index(drop=True)
    
    return cleaned_df

def validate_data(df, required_columns):
    """
    Validate that the DataFrame contains all required columns and has no empty rows.
    
    Args:
        df (pd.DataFrame): DataFrame to validate.
        required_columns (list): List of required column names.
    
    Returns:
        tuple: (bool, str) indicating validation success and message.
    """
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        return False, f"Missing required columns: {missing_columns}"
    
    if df.empty:
        return False, "DataFrame is empty"
    
    return True, "Data validation passed"