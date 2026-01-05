
import pandas as pd

def clean_dataset(df, column_mapping=None, remove_duplicates=True):
    """
    Clean a pandas DataFrame by standardizing column names and removing duplicates.
    
    Args:
        df: pandas DataFrame to clean
        column_mapping: Optional dictionary mapping old column names to new names
        remove_duplicates: Boolean indicating whether to remove duplicate rows
    
    Returns:
        Cleaned pandas DataFrame
    """
    cleaned_df = df.copy()
    
    if column_mapping:
        cleaned_df = cleaned_df.rename(columns=column_mapping)
    
    cleaned_df.columns = cleaned_df.columns.str.strip().str.lower().str.replace(' ', '_')
    
    if remove_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    cleaned_df = cleaned_df.reset_index(drop=True)
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate that a DataFrame meets basic requirements.
    
    Args:
        df: pandas DataFrame to validate
        required_columns: List of column names that must be present
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if df.empty:
        return False, "DataFrame is empty"
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
    
    return True, "DataFrame is valid"

def sample_data_cleaning():
    """
    Example usage of the data cleaning functions.
    """
    data = {
        'Customer ID': [1, 2, 3, 1, 4],
        'First Name': ['John', 'Jane', 'Bob', 'John', 'Alice'],
        'Last Name': ['Doe', 'Smith', 'Johnson', 'Doe', 'Brown'],
        'Email': ['john@example.com', 'jane@example.com', 'bob@example.com', 
                  'john@example.com', 'alice@example.com']
    }
    
    df = pd.DataFrame(data)
    
    column_mapping = {
        'Customer ID': 'customer_id',
        'First Name': 'first_name',
        'Last Name': 'last_name',
        'Email': 'email'
    }
    
    cleaned_df = clean_dataset(df, column_mapping=column_mapping)
    
    is_valid, message = validate_dataframe(cleaned_df, 
                                          required_columns=['customer_id', 'email'])
    
    print(f"Original shape: {df.shape}")
    print(f"Cleaned shape: {cleaned_df.shape}")
    print(f"Validation: {message}")
    print(f"Cleaned columns: {list(cleaned_df.columns)}")
    
    return cleaned_df

if __name__ == "__main__":
    cleaned_data = sample_data_cleaning()
    print("\nCleaned data preview:")
    print(cleaned_data.head())