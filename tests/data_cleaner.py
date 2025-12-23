
def remove_duplicates(data_list):
    """
    Remove duplicate entries from a list while preserving order.
    
    Args:
        data_list: A list of elements (must be hashable).
    
    Returns:
        A new list with duplicates removed.
    """
    seen = set()
    result = []
    for item in data_list:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result

def clean_data_with_key(data_list, key_func=None):
    """
    Remove duplicates based on a key function.
    
    Args:
        data_list: A list of elements.
        key_func: A function to extract comparison key (default: identity).
    
    Returns:
        A new list with duplicates removed based on key.
    """
    if key_func is None:
        return remove_duplicates(data_list)
    
    seen = set()
    result = []
    for item in data_list:
        key = key_func(item)
        if key not in seen:
            seen.add(key)
            result.append(item)
    return result

if __name__ == "__main__":
    # Example usage
    sample_data = [1, 2, 2, 3, 4, 4, 5, 1]
    cleaned = remove_duplicates(sample_data)
    print(f"Original: {sample_data}")
    print(f"Cleaned: {cleaned}")
    
    # Example with key function
    data_dicts = [
        {"id": 1, "name": "Alice"},
        {"id": 2, "name": "Bob"},
        {"id": 1, "name": "Alice"},
        {"id": 3, "name": "Charlie"}
    ]
    cleaned_dicts = clean_data_with_key(data_dicts, key_func=lambda x: x["id"])
    print(f"\nOriginal dicts: {data_dicts}")
    print(f"Cleaned dicts: {cleaned_dicts}")
import pandas as pd
import re

def clean_dataset(df, column_names):
    """
    Clean a pandas DataFrame by removing duplicates and normalizing text columns.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean
        column_names (list): List of column names to normalize
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    # Remove duplicate rows
    df_cleaned = df.drop_duplicates().reset_index(drop=True)
    
    # Normalize specified text columns
    for col in column_names:
        if col in df_cleaned.columns:
            df_cleaned[col] = df_cleaned[col].apply(
                lambda x: re.sub(r'\s+', ' ', str(x).strip().lower()) 
                if pd.notnull(x) else x
            )
    
    return df_cleaned

def validate_email_column(df, email_column):
    """
    Validate email addresses in a specified column.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        email_column (str): Name of the column containing email addresses
    
    Returns:
        pd.DataFrame: DataFrame with validation results
    """
    if email_column not in df.columns:
        raise ValueError(f"Column '{email_column}' not found in DataFrame")
    
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    
    df['email_valid'] = df[email_column].apply(
        lambda x: bool(re.match(email_pattern, str(x))) if pd.notnull(x) else False
    )
    
    return df

def main():
    # Example usage
    sample_data = {
        'name': ['John Doe', 'Jane Smith', 'John Doe', 'Bob Johnson', ''],
        'email': ['john@example.com', 'jane@test.org', 'invalid-email', 'bob@company.co', None],
        'age': [25, 30, 25, 35, 40]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    # Clean the dataset
    cleaned_df = clean_dataset(df, ['name'])
    print("Cleaned DataFrame:")
    print(cleaned_df)
    print("\n" + "="*50 + "\n")
    
    # Validate emails
    validated_df = validate_email_column(cleaned_df, 'email')
    print("DataFrame with email validation:")
    print(validated_df[['name', 'email', 'email_valid']])

if __name__ == "__main__":
    main()import pandas as pd

def clean_dataset(df):
    """
    Clean a pandas DataFrame by removing null values and duplicate rows.
    
    Args:
        df (pd.DataFrame): The input DataFrame to be cleaned.
    
    Returns:
        pd.DataFrame: The cleaned DataFrame.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    
    cleaned_df = df.copy()
    
    cleaned_df = cleaned_df.dropna()
    
    cleaned_df = cleaned_df.drop_duplicates()
    
    cleaned_df = cleaned_df.reset_index(drop=True)
    
    return cleaned_df

def validate_data(df, required_columns):
    """
    Validate that the DataFrame contains all required columns.
    
    Args:
        df (pd.DataFrame): The DataFrame to validate.
        required_columns (list): List of column names that must be present.
    
    Returns:
        bool: True if all required columns are present, False otherwise.
    """
    if not isinstance(df, pd.DataFrame):
        return False
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        print(f"Missing columns: {missing_columns}")
        return False
    
    return True

if __name__ == "__main__":
    sample_data = {
        'name': ['Alice', 'Bob', 'Charlie', None, 'Alice'],
        'age': [25, 30, None, 35, 25],
        'city': ['NYC', 'LA', 'Chicago', 'Boston', 'NYC']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n")
    
    cleaned_df = clean_dataset(df)
    print("Cleaned DataFrame:")
    print(cleaned_df)
    
    required_cols = ['name', 'age']
    is_valid = validate_data(cleaned_df, required_cols)
    print(f"\nData validation result: {is_valid}")