
import re

def clean_string(text):
    """
    Clean and normalize a string by:
    - Removing leading/trailing whitespace
    - Converting to lowercase
    - Removing extra spaces between words
    - Removing non-alphanumeric characters except basic punctuation
    """
    if not isinstance(text, str):
        return text

    # Remove leading/trailing whitespace
    text = text.strip()

    # Convert to lowercase
    text = text.lower()

    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)

    # Remove non-alphanumeric characters except spaces, periods, commas, and hyphens
    text = re.sub(r'[^a-z0-9\s.,-]', '', text)

    return text

def normalize_phone_number(phone):
    """
    Normalize a phone number string by removing all non-digit characters.
    Returns the cleaned digits or None if no digits are found.
    """
    if not isinstance(phone, str):
        return None

    digits = re.sub(r'\D', '', phone)

    return digits if digits else None

def validate_email(email):
    """
    Basic email validation using a regular expression.
    Returns True if the email format is valid, False otherwise.
    """
    if not isinstance(email, str):
        return False

    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to clean
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
    return filtered_df

def calculate_summary_statistics(df, column):
    """
    Calculate summary statistics for a DataFrame column.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to analyze
    
    Returns:
    dict: Dictionary containing summary statistics
    """
    stats = {
        'mean': df[column].mean(),
        'median': df[column].median(),
        'std': df[column].std(),
        'min': df[column].min(),
        'max': df[column].max(),
        'count': df[column].count()
    }
    
    return stats

def clean_numeric_data(df, columns):
    """
    Clean multiple numeric columns by removing outliers.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    columns (list): List of column names to clean
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    for column in columns:
        if column in cleaned_df.columns and pd.api.types.is_numeric_dtype(cleaned_df[column]):
            cleaned_df = remove_outliers_iqr(cleaned_df, column)
    
    return cleaned_df