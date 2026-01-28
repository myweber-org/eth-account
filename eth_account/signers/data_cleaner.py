
import pandas as pd
import re

def clean_dataframe(df, text_columns=None):
    """
    Clean a DataFrame by removing duplicate rows and standardizing text in specified columns.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        text_columns (list, optional): List of column names containing text to standardize.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    # Remove duplicate rows
    initial_rows = cleaned_df.shape[0]
    cleaned_df = cleaned_df.drop_duplicates()
    removed_duplicates = initial_rows - cleaned_df.shape[0]
    
    # Standardize text in specified columns
    if text_columns:
        for col in text_columns:
            if col in cleaned_df.columns:
                cleaned_df[col] = cleaned_df[col].apply(_standardize_text)
    
    return cleaned_df

def _standardize_text(text):
    """
    Standardize text by converting to lowercase, removing extra whitespace,
    and stripping special characters.
    
    Args:
        text (str): Input text to standardize.
    
    Returns:
        str: Standardized text.
    """
    if not isinstance(text, str):
        return text
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove special characters except alphanumeric and spaces
    text = re.sub(r'[^a-z0-9\s]', '', text)
    
    return text

def validate_email_column(df, email_column):
    """
    Validate email addresses in a DataFrame column.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        email_column (str): Name of the column containing email addresses.
    
    Returns:
        pd.DataFrame: DataFrame with additional 'email_valid' column.
    """
    if email_column not in df.columns:
        raise ValueError(f"Column '{email_column}' not found in DataFrame")
    
    validated_df = df.copy()
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    
    validated_df['email_valid'] = validated_df[email_column].apply(
        lambda x: bool(re.match(email_pattern, str(x))) if pd.notnull(x) else False
    )
    
    return validated_dfimport pandas as pd

def clean_dataset(df):
    """
    Remove null values and duplicate rows from a pandas DataFrame.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to be cleaned.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame with nulls removed and duplicates dropped.
    """
    # Remove rows with any null values
    df_cleaned = df.dropna()
    
    # Remove duplicate rows
    df_cleaned = df_cleaned.drop_duplicates()
    
    # Reset index after cleaning
    df_cleaned = df_cleaned.reset_index(drop=True)
    
    return df_cleaned

def validate_dataframe(df):
    """
    Validate that the input is a pandas DataFrame and not empty.
    
    Parameters:
    df: Input to validate
    
    Returns:
    bool: True if valid, False otherwise
    """
    if not isinstance(df, pd.DataFrame):
        print("Error: Input must be a pandas DataFrame")
        return False
    
    if df.empty:
        print("Warning: DataFrame is empty")
        return False
    
    return True

# Example usage (commented out for production)
# if __name__ == "__main__":
#     sample_data = pd.DataFrame({
#         'A': [1, 2, None, 4, 2],
#         'B': [5, 6, 7, None, 6],
#         'C': [8, 9, 10, 11, 9]
#     })
#     
#     if validate_dataframe(sample_data):
#         cleaned_data = clean_dataset(sample_data)
#         print("Original shape:", sample_data.shape)
#         print("Cleaned shape:", cleaned_data.shape)
#         print("\nCleaned data:")
#         print(cleaned_data)