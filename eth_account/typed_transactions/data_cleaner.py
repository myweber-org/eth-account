
import pandas as pd
import re

def clean_dataframe(df, columns_to_clean=None):
    """
    Clean a pandas DataFrame by removing duplicate rows and normalizing
    string columns (strip whitespace, convert to lowercase).
    """
    # Remove duplicate rows
    df_cleaned = df.drop_duplicates().reset_index(drop=True)
    
    # If specific columns are provided, clean only those columns
    # Otherwise, clean all object (string) columns
    if columns_to_clean is None:
        columns_to_clean = df_cleaned.select_dtypes(include=['object']).columns
    
    for col in columns_to_clean:
        if col in df_cleaned.columns and df_cleaned[col].dtype == 'object':
            # Strip whitespace and convert to lowercase
            df_cleaned[col] = df_cleaned[col].astype(str).str.strip().str.lower()
            # Replace multiple spaces with single space
            df_cleaned[col] = df_cleaned[col].apply(lambda x: re.sub(r'\s+', ' ', x))
    
    return df_cleaned

def validate_email_column(df, email_column):
    """
    Validate email addresses in a specified column.
    Returns a DataFrame with valid emails and a separate Series of invalid ones.
    """
    if email_column not in df.columns:
        raise ValueError(f"Column '{email_column}' not found in DataFrame")
    
    # Simple email regex pattern
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    
    # Create mask for valid emails
    valid_mask = df[email_column].astype(str).str.match(email_pattern)
    
    valid_emails = df[valid_mask].copy()
    invalid_emails = df[~valid_mask][email_column]
    
    return valid_emails, invalid_emails

def remove_outliers_iqr(df, column, multiplier=1.5):
    """
    Remove outliers from a numeric column using the Interquartile Range (IQR) method.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    if not pd.api.types.is_numeric_dtype(df[column]):
        raise ValueError(f"Column '{column}' is not numeric")
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    # Filter rows where column values are within bounds
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)].copy()
    
    return filtered_df