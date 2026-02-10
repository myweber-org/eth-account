
import pandas as pd
import re

def clean_dataframe(df, column_mapping=None, drop_duplicates=True, normalize_text=True):
    """
    Clean a pandas DataFrame by removing duplicates and normalizing text columns.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        column_mapping (dict, optional): Dictionary mapping original column names to new names.
        drop_duplicates (bool, optional): Whether to drop duplicate rows. Defaults to True.
        normalize_text (bool, optional): Whether to normalize text columns. Defaults to True.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    df_clean = df.copy()
    
    if column_mapping:
        df_clean = df_clean.rename(columns=column_mapping)
    
    if drop_duplicates:
        df_clean = df_clean.drop_duplicates().reset_index(drop=True)
    
    if normalize_text:
        for col in df_clean.select_dtypes(include=['object']).columns:
            df_clean[col] = df_clean[col].apply(lambda x: normalize_string(x) if isinstance(x, str) else x)
    
    return df_clean

def normalize_string(text):
    """
    Normalize a string by converting to lowercase, removing extra whitespace,
    and stripping special characters.
    
    Args:
        text (str): Input string to normalize.
    
    Returns:
        str: Normalized string.
    """
    if not isinstance(text, str):
        return text
    
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    text = re.sub(r'[^\w\s-]', '', text)
    
    return text

def validate_email(email):
    """
    Validate an email address format.
    
    Args:
        email (str): Email address to validate.
    
    Returns:
        bool: True if email format is valid, False otherwise.
    """
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email)) if isinstance(email, str) else False

def main():
    sample_data = {
        'Name': ['John Doe', 'Jane Smith', 'John Doe', 'Alice Johnson  '],
        'Email': ['john@example.com', 'jane@example.com', 'invalid-email', 'alice@example.com'],
        'Age': [25, 30, 25, 28]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n")
    
    cleaned_df = clean_dataframe(df, column_mapping={'Name': 'Full Name'})
    print("Cleaned DataFrame:")
    print(cleaned_df)
    print("\n")
    
    cleaned_df['Email_Valid'] = cleaned_df['Email'].apply(validate_email)
    print("DataFrame with Email Validation:")
    print(cleaned_df)

if __name__ == "__main__":
    main()