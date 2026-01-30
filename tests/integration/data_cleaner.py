
import pandas as pd
import re

def clean_dataset(df, column_mapping=None, drop_duplicates=True, normalize_text=True):
    """
    Clean a pandas DataFrame by removing duplicates and normalizing text columns.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean
        column_mapping (dict, optional): Dictionary mapping original column names to new names
        drop_duplicates (bool): Whether to remove duplicate rows
        normalize_text (bool): Whether to normalize text columns (strip whitespace, lowercase)
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    if column_mapping:
        cleaned_df = cleaned_df.rename(columns=column_mapping)
    
    if drop_duplicates:
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        removed = initial_rows - len(cleaned_df)
        print(f"Removed {removed} duplicate rows")
    
    if normalize_text:
        for column in cleaned_df.select_dtypes(include=['object']).columns:
            cleaned_df[column] = cleaned_df[column].astype(str).str.strip().str.lower()
            cleaned_df[column] = cleaned_df[column].apply(
                lambda x: re.sub(r'\s+', ' ', x) if isinstance(x, str) else x
            )
    
    cleaned_df = cleaned_df.reset_index(drop=True)
    
    missing_values = cleaned_df.isnull().sum().sum()
    if missing_values > 0:
        print(f"Warning: Dataset contains {missing_values} missing values")
    
    return cleaned_df

def validate_email_column(df, email_column='email'):
    """
    Validate email addresses in a specified column.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        email_column (str): Name of the column containing email addresses
    
    Returns:
        pd.DataFrame: DataFrame with additional 'email_valid' column
    """
    if email_column not in df.columns:
        raise ValueError(f"Column '{email_column}' not found in DataFrame")
    
    validated_df = df.copy()
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    
    validated_df['email_valid'] = validated_df[email_column].apply(
        lambda x: bool(re.match(email_pattern, str(x))) if pd.notnull(x) else False
    )
    
    valid_count = validated_df['email_valid'].sum()
    total_count = len(validated_df)
    print(f"Valid emails: {valid_count}/{total_count} ({valid_count/total_count*100:.1f}%)")
    
    return validated_df

def sample_data_for_testing():
    """Create sample data for testing the cleaning functions."""
    data = {
        'name': ['John Doe', 'Jane Smith', 'John Doe', 'Bob Johnson', 'Alice Brown  '],
        'email': ['john@example.com', 'jane@example.com', 'JOHN@example.com', 'invalid-email', 'alice@test.org'],
        'age': [25, 30, 25, 35, 28],
        'city': ['New York', 'Los Angeles', 'New York', 'Chicago', 'Boston']
    }
    
    return pd.DataFrame(data)

if __name__ == "__main__":
    df = sample_data_for_testing()
    print("Original dataset:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    cleaned = clean_dataset(df, drop_duplicates=True, normalize_text=True)
    print("Cleaned dataset:")
    print(cleaned)
    print("\n" + "="*50 + "\n")
    
    validated = validate_email_column(cleaned, 'email')
    print("Dataset with email validation:")
    print(validated[['name', 'email', 'email_valid']])