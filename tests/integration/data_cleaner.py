
import pandas as pd
import re

def clean_dataframe(df, columns_to_clean=None):
    """
    Clean a pandas DataFrame by removing duplicates and normalizing string columns.
    """
    cleaned_df = df.copy()
    
    # Remove duplicate rows
    cleaned_df = cleaned_df.drop_duplicates()
    
    # If specific columns are provided, clean only those; otherwise clean all object columns
    if columns_to_clean is None:
        columns_to_clean = cleaned_df.select_dtypes(include=['object']).columns
    else:
        columns_to_clean = [col for col in columns_to_clean if col in cleaned_df.columns]
    
    for column in columns_to_clean:
        if cleaned_df[column].dtype == 'object':
            cleaned_df[column] = cleaned_df[column].apply(_normalize_string)
    
    return cleaned_df

def _normalize_string(value):
    """
    Normalize a string: lowercase, strip whitespace, and remove extra spaces.
    """
    if isinstance(value, str):
        value = value.lower()
        value = value.strip()
        value = re.sub(r'\s+', ' ', value)
    return value

def validate_email_column(df, email_column):
    """
    Validate email addresses in a specified column.
    Returns a Series with boolean values indicating valid emails.
    """
    if email_column not in df.columns:
        raise ValueError(f"Column '{email_column}' not found in DataFrame")
    
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return df[email_column].apply(lambda x: bool(re.match(email_pattern, str(x))))