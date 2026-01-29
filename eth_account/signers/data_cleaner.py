import pandas as pd
import numpy as np

def remove_duplicates(df, subset=None, keep='first'):
    """
    Remove duplicate rows from a DataFrame.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    subset (list, optional): Columns to consider for duplicates
    keep (str, optional): Which duplicates to keep - 'first', 'last', or False
    
    Returns:
    pd.DataFrame: DataFrame with duplicates removed
    """
    if df.empty:
        return df
    
    cleaned_df = df.drop_duplicates(subset=subset, keep=keep)
    
    removed_count = len(df) - len(cleaned_df)
    print(f"Removed {removed_count} duplicate rows")
    
    return cleaned_df

def clean_missing_values(df, strategy='drop', fill_value=None):
    """
    Handle missing values in DataFrame.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    strategy (str): 'drop' to remove rows, 'fill' to fill values
    fill_value: Value to fill missing values with
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    if strategy == 'drop':
        cleaned_df = df.dropna()
        print(f"Removed {len(df) - len(cleaned_df)} rows with missing values")
    elif strategy == 'fill':
        if fill_value is not None:
            cleaned_df = df.fillna(fill_value)
        else:
            cleaned_df = df.fillna(df.mean())
        print("Filled missing values")
    
    return cleaned_df

def validate_data(df, required_columns=None, min_rows=1):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate
    required_columns (list): List of required column names
    min_rows (int): Minimum number of rows required
    
    Returns:
    bool: True if validation passes
    """
    if df.empty:
        print("DataFrame is empty")
        return False
    
    if len(df) < min_rows:
        print(f"DataFrame has fewer than {min_rows} rows")
        return False
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            print(f"Missing required columns: {missing_cols}")
            return False
    
    return True

def process_dataframe(df, operations=None):
    """
    Apply multiple cleaning operations to DataFrame.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    operations (list): List of operations to apply
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    if operations is None:
        operations = ['remove_duplicates', 'clean_missing']
    
    cleaned_df = df.copy()
    
    for operation in operations:
        if operation == 'remove_duplicates':
            cleaned_df = remove_duplicates(cleaned_df)
        elif operation == 'clean_missing':
            cleaned_df = clean_missing_values(cleaned_df)
    
    return cleaned_df

if __name__ == "__main__":
    sample_data = {
        'id': [1, 2, 3, 1, 4, 5, 3],
        'name': ['Alice', 'Bob', 'Charlie', 'Alice', 'David', 'Eve', 'Charlie'],
        'age': [25, 30, None, 25, 35, 28, 32],
        'score': [85.5, 92.0, 78.5, 85.5, 88.0, 95.5, 78.5]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nShape:", df.shape)
    
    cleaned_df = process_dataframe(df)
    print("\nCleaned DataFrame:")
    print(cleaned_df)
    print("\nShape:", cleaned_df.shape)import pandas as pd
import re

def clean_dataframe(df, text_column):
    """
    Clean a DataFrame by removing duplicate rows and standardizing text in a specified column.
    """
    # Remove duplicate rows
    df_clean = df.drop_duplicates().reset_index(drop=True)
    
    # Standardize text: lowercase and remove extra whitespace
    if text_column in df_clean.columns:
        df_clean[text_column] = df_clean[text_column].apply(
            lambda x: re.sub(r'\s+', ' ', str(x).strip().lower())
        )
    
    return df_clean

def filter_by_keyword(df, text_column, keyword):
    """
    Filter rows where the specified text column contains a given keyword.
    """
    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in DataFrame")
    
    filtered_df = df[df[text_column].str.contains(keyword, case=False, na=False)]
    return filtered_df.reset_index(drop=True)

# Example usage (commented out)
# if __name__ == "__main__":
#     data = {'ID': [1, 2, 3, 4, 1],
#             'Text': ['Hello World', 'hello world', 'Test   data', 'Another test', 'Hello World']}
#     df = pd.DataFrame(data)
#     cleaned = clean_dataframe(df, 'Text')
#     print("Cleaned DataFrame:")
#     print(cleaned)
#     filtered = filter_by_keyword(cleaned, 'Text', 'test')
#     print("\nFiltered by 'test':")
#     print(filtered)