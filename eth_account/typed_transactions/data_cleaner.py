
import pandas as pd
import re

def clean_text_column(df, column_name):
    """Standardize text by converting to lowercase and removing extra whitespace."""
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame")
    
    df[column_name] = df[column_name].astype(str).str.lower()
    df[column_name] = df[column_name].apply(lambda x: re.sub(r'\s+', ' ', x).strip())
    return df

def remove_duplicates(df, subset=None, keep='first'):
    """Remove duplicate rows from DataFrame."""
    return df.drop_duplicates(subset=subset, keep=keep)

def validate_email_column(df, column_name):
    """Validate email format in specified column."""
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame")
    
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    df['is_valid_email'] = df[column_name].apply(lambda x: bool(re.match(email_pattern, str(x))))
    return df

def process_dataframe(df, text_columns=None, email_column=None, deduplicate=True):
    """Main function to clean and process DataFrame."""
    if text_columns:
        for col in text_columns:
            df = clean_text_column(df, col)
    
    if email_column:
        df = validate_email_column(df, email_column)
    
    if deduplicate:
        df = remove_duplicates(df)
    
    return df