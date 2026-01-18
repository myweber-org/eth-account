
import pandas as pd

def clean_dataframe(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean
    drop_duplicates (bool): Whether to drop duplicate rows
    fill_missing (str): Method to fill missing values - 'mean', 'median', or 'drop'
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    if fill_missing == 'drop':
        cleaned_df = cleaned_df.dropna()
    elif fill_missing in ['mean', 'median']:
        numeric_cols = cleaned_df.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            if fill_missing == 'mean':
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mean())
            else:
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate
    required_columns (list): List of required column names
    
    Returns:
    tuple: (is_valid, error_message)
    """
    if df.empty:
        return False, "DataFrame is empty"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    return True, "DataFrame is valid"
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
    
    if column_mapping:
        cleaned_df = cleaned_df.rename(columns=column_mapping)
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates().reset_index(drop=True)
    
    if normalize_text:
        for col in cleaned_df.select_dtypes(include=['object']).columns:
            cleaned_df[col] = cleaned_df[col].astype(str).str.strip().str.lower()
    
    return cleaned_df

def remove_special_characters(text, keep_pattern=r'[a-zA-Z0-9\s]'):
    """
    Remove special characters from text, keeping only specified pattern.
    
    Args:
        text (str): Input text
        keep_pattern (str): Regex pattern of characters to keep
    
    Returns:
        str: Cleaned text
    """
    if pd.isna(text):
        return text
    return re.sub(f'[^{keep_pattern}]', '', str(text))

def validate_email(email):
    """
    Validate email format using regex pattern.
    
    Args:
        email (str): Email address to validate
    
    Returns:
        bool: True if email is valid, False otherwise
    """
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, str(email))) if pd.notna(email) else False

def fill_missing_values(df, strategy='mean', fill_value=None):
    """
    Fill missing values in numeric columns using specified strategy.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        strategy (str): 'mean', 'median', 'mode', or 'constant'
        fill_value: Value to use when strategy is 'constant'
    
    Returns:
        pd.DataFrame: DataFrame with missing values filled
    """
    df_filled = df.copy()
    
    numeric_cols = df_filled.select_dtypes(include=['number']).columns
    
    for col in numeric_cols:
        if strategy == 'mean':
            df_filled[col] = df_filled[col].fillna(df_filled[col].mean())
        elif strategy == 'median':
            df_filled[col] = df_filled[col].fillna(df_filled[col].median())
        elif strategy == 'mode':
            df_filled[col] = df_filled[col].fillna(df_filled[col].mode()[0])
        elif strategy == 'constant' and fill_value is not None:
            df_filled[col] = df_filled[col].fillna(fill_value)
    
    return df_filled

if __name__ == "__main__":
    sample_data = {
        'name': ['John Doe', 'Jane Smith', 'John Doe', ' Bob Johnson ', None],
        'email': ['john@example.com', 'jane@test.org', 'invalid-email', 'bob@company.net', None],
        'age': [25, 30, 25, 35, None],
        'score': [85.5, 92.0, 85.5, 78.3, None]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nCleaned DataFrame:")
    cleaned = clean_dataframe(df, drop_duplicates=True, normalize_text=True)
    print(cleaned)
    
    cleaned['email_valid'] = cleaned['email'].apply(validate_email)
    print("\nDataFrame with email validation:")
    print(cleaned[['email', 'email_valid']])
    
    filled_df = fill_missing_values(cleaned, strategy='mean')
    print("\nDataFrame with filled missing values:")
    print(filled_df)