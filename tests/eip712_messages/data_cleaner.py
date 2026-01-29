
import pandas as pd
import re

def clean_dataframe(df, column_mapping=None, drop_duplicates=True, normalize_text=True):
    """
    Clean a pandas DataFrame by removing duplicates and normalizing text columns.
    
    Args:
        df: pandas DataFrame to clean
        column_mapping: Optional dictionary to rename columns
        drop_duplicates: Whether to remove duplicate rows
        normalize_text: Whether to normalize text columns (strip, lowercase, remove extra spaces)
    
    Returns:
        Cleaned pandas DataFrame
    """
    cleaned_df = df.copy()
    
    if column_mapping:
        cleaned_df = cleaned_df.rename(columns=column_mapping)
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates().reset_index(drop=True)
    
    if normalize_text:
        for column in cleaned_df.select_dtypes(include=['object']).columns:
            cleaned_df[column] = cleaned_df[column].apply(
                lambda x: re.sub(r'\s+', ' ', str(x).strip().lower()) if pd.notna(x) else x
            )
    
    return cleaned_df

def validate_email(email):
    """
    Validate email format using regex pattern.
    
    Args:
        email: String email to validate
    
    Returns:
        Boolean indicating if email is valid
    """
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, str(email))) if pd.notna(email) else False

def filter_valid_emails(df, email_column):
    """
    Filter DataFrame to only include rows with valid email addresses.
    
    Args:
        df: pandas DataFrame
        email_column: Name of column containing email addresses
    
    Returns:
        DataFrame with only valid email rows
    """
    mask = df[email_column].apply(validate_email)
    return df[mask].reset_index(drop=True)