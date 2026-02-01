import pandas as pd
import re

def clean_dataframe(df, text_columns=None, drop_duplicates=True, lowercase=True, strip_whitespace=True):
    """
    Clean a pandas DataFrame by removing duplicates and standardizing text columns.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        text_columns (list, optional): List of column names to apply text cleaning.
            If None, all object dtype columns are cleaned.
        drop_duplicates (bool): Whether to remove duplicate rows.
        lowercase (bool): Convert text to lowercase.
        strip_whitespace (bool): Remove leading/trailing whitespace.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        removed = initial_rows - len(cleaned_df)
        print(f"Removed {removed} duplicate rows.")
    
    if text_columns is None:
        text_columns = cleaned_df.select_dtypes(include=['object']).columns.tolist()
    
    for col in text_columns:
        if col in cleaned_df.columns:
            if lowercase:
                cleaned_df[col] = cleaned_df[col].astype(str).str.lower()
            if strip_whitespace:
                cleaned_df[col] = cleaned_df[col].astype(str).str.strip()
    
    return cleaned_df

def remove_special_characters(text, keep_pattern=r'[^a-zA-Z0-9\s]'):
    """
    Remove special characters from text.
    
    Args:
        text (str): Input text.
        keep_pattern (str): Regex pattern of characters to keep.
    
    Returns:
        str: Cleaned text.
    """
    if pd.isna(text):
        return text
    return re.sub(keep_pattern, '', str(text))

def validate_email(email):
    """
    Validate email format using regex.
    
    Args:
        email (str): Email address to validate.
    
    Returns:
        bool: True if email format is valid.
    """
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    if pd.isna(email):
        return False
    return bool(re.match(pattern, str(email)))