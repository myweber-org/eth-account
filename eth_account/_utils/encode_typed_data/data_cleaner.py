
import pandas as pd
import re

def clean_dataframe(df, column_mapping=None, drop_duplicates=True, normalize_text=True):
    """
    Clean a pandas DataFrame by removing duplicates and normalizing text columns.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean
        column_mapping (dict, optional): Dictionary mapping old column names to new ones
        drop_duplicates (bool): Whether to remove duplicate rows
        normalize_text (bool): Whether to normalize text columns (strip, lower case)
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    # Rename columns if mapping provided
    if column_mapping:
        cleaned_df = cleaned_df.rename(columns=column_mapping)
    
    # Remove duplicate rows
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates().reset_index(drop=True)
    
    # Normalize text columns
    if normalize_text:
        text_columns = cleaned_df.select_dtypes(include=['object']).columns
        for col in text_columns:
            cleaned_df[col] = cleaned_df[col].astype(str).str.strip().str.lower()
    
    return cleaned_df

def validate_email(email_string):
    """
    Validate email format using regex pattern.
    
    Args:
        email_string (str): Email address to validate
    
    Returns:
        bool: True if email format is valid, False otherwise
    """
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, str(email_string)))

def filter_valid_emails(df, email_column):
    """
    Filter DataFrame to only include rows with valid email addresses.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        email_column (str): Name of column containing email addresses
    
    Returns:
        pd.DataFrame: DataFrame with only valid email rows
    """
    mask = df[email_column].apply(validate_email)
    return df[mask].reset_index(drop=True)

def remove_special_characters(text, keep_chars=''):
    """
    Remove special characters from text, keeping only alphanumeric and specified characters.
    
    Args:
        text (str): Input text
        keep_chars (str): Additional characters to keep (e.g., ' .-_')
    
    Returns:
        str: Cleaned text
    """
    pattern = f'[^a-zA-Z0-9{re.escape(keep_chars)}]'
    return re.sub(pattern, '', str(text))