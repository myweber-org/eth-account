import pandas as pd
import re

def clean_dataframe(df, text_column):
    """
    Remove duplicate rows and standardize text in specified column.
    """
    # Remove duplicates
    df_clean = df.drop_duplicates().reset_index(drop=True)
    
    # Standardize text: lowercase, remove extra spaces
    df_clean[text_column] = df_clean[text_column].apply(
        lambda x: re.sub(r'\s+', ' ', str(x).strip().lower())
    )
    
    return df_clean

def validate_email_column(df, email_column):
    """
    Validate email format in specified column.
    """
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    df['valid_email'] = df[email_column].apply(
        lambda x: bool(re.match(email_pattern, str(x)))
    )
    return df

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'name': ['John Doe', 'Jane Smith', 'John Doe', 'Bob Johnson'],
        'email': ['john@example.com', 'jane@example.com', 'invalid-email', 'bob@example.org']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    
    df_clean = clean_dataframe(df, 'name')
    df_validated = validate_email_column(df_clean, 'email')
    
    print("\nCleaned and Validated DataFrame:")
    print(df_validated)