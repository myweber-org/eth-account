import pandas as pd
import re

def clean_dataframe(df, columns_to_clean=None):
    """
    Clean a pandas DataFrame by removing duplicates and normalizing string columns.
    """
    cleaned_df = df.copy()
    
    # Remove duplicate rows
    initial_rows = cleaned_df.shape[0]
    cleaned_df.drop_duplicates(inplace=True)
    removed_duplicates = initial_rows - cleaned_df.shape[0]
    
    # Normalize string columns
    if columns_to_clean is None:
        # Automatically detect string columns
        string_columns = cleaned_df.select_dtypes(include=['object']).columns
    else:
        string_columns = [col for col in columns_to_clean if col in cleaned_df.columns]
    
    for col in string_columns:
        if cleaned_df[col].dtype == 'object':
            cleaned_df[col] = cleaned_df[col].apply(normalize_string)
    
    return cleaned_df, removed_duplicates

def normalize_string(text):
    """
    Normalize a string by converting to lowercase, removing extra whitespace,
    and stripping special characters.
    """
    if pd.isna(text):
        return text
    
    # Convert to string if not already
    text = str(text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    # Remove special characters (keep alphanumeric and basic punctuation)
    text = re.sub(r'[^\w\s.,!?-]', '', text)
    
    return text.strip()

def validate_dataframe(df, required_columns=None):
    """
    Validate a DataFrame by checking for required columns and null values.
    """
    validation_results = {
        'is_valid': True,
        'missing_columns': [],
        'null_counts': {}
    }
    
    if required_columns:
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            validation_results['missing_columns'] = missing
            validation_results['is_valid'] = False
    
    # Check for null values in each column
    for col in df.columns:
        null_count = df[col].isnull().sum()
        if null_count > 0:
            validation_results['null_counts'][col] = null_count
    
    return validation_results

# Example usage (commented out for production)
# if __name__ == "__main__":
#     # Create sample data
#     data = {
#         'name': ['John Doe', 'Jane Smith', 'John Doe', 'Bob Johnson  ', 'ALICE'],
#         'email': ['john@example.com', 'jane@example.com', 'john@example.com', 'bob@example.com', 'alice@example.com'],
#         'age': [25, 30, 25, 35, 28]
#     }
#     
#     df = pd.DataFrame(data)
#     cleaned_df, duplicates_removed = clean_dataframe(df)
#     print(f"Removed {duplicates_removed} duplicate rows")
#     print(cleaned_df)
#     
#     validation = validate_dataframe(cleaned_df, required_columns=['name', 'email', 'age'])
#     print(f"Validation results: {validation}")