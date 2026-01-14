
import pandas as pd
import re

def clean_text_column(df, column_name):
    """
    Standardize text in a DataFrame column by converting to lowercase,
    removing extra whitespace, and stripping special characters.
    """
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame")
    
    df[column_name] = df[column_name].astype(str)
    df[column_name] = df[column_name].str.lower()
    df[column_name] = df[column_name].str.strip()
    df[column_name] = df[column_name].apply(lambda x: re.sub(r'[^\w\s]', '', x))
    df[column_name] = df[column_name].apply(lambda x: re.sub(r'\s+', ' ', x))
    
    return df

def remove_duplicate_rows(df, subset=None, keep='first'):
    """
    Remove duplicate rows from DataFrame based on specified columns.
    """
    return df.drop_duplicates(subset=subset, keep=keep)

def validate_email_column(df, column_name):
    """
    Validate email addresses in a DataFrame column using regex pattern.
    Returns a boolean Series indicating valid emails.
    """
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame")
    
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return df[column_name].str.match(pattern)

def main():
    # Example usage
    data = {
        'name': ['John Doe', 'Jane Smith', 'John Doe', 'Bob Johnson'],
        'email': ['john@example.com', 'jane@example.com', 'invalid-email', 'bob@example.com'],
        'notes': ['  Some text here!  ', 'Another note.', '  Some text here!  ', 'Final note.']
    }
    
    df = pd.DataFrame(data)
    print("Original DataFrame:")
    print(df)
    print()
    
    # Clean text column
    df = clean_text_column(df, 'notes')
    print("After cleaning 'notes' column:")
    print(df)
    print()
    
    # Remove duplicates
    df = remove_duplicate_rows(df, subset=['name', 'notes'])
    print("After removing duplicates:")
    print(df)
    print()
    
    # Validate emails
    valid_emails = validate_email_column(df, 'email')
    print("Email validation results:")
    print(valid_emails)

if __name__ == "__main__":
    main()