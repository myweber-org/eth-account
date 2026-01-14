
import pandas as pd

def remove_duplicates(df, subset=None, keep='first'):
    """
    Remove duplicate rows from a DataFrame.
    
    Args:
        df: pandas DataFrame
        subset: column label or sequence of labels to consider for duplicates
        keep: determines which duplicates to keep ('first', 'last', False)
    
    Returns:
        DataFrame with duplicates removed
    """
    if df.empty:
        return df
    
    cleaned_df = df.drop_duplicates(subset=subset, keep=keep)
    
    removed_count = len(df) - len(cleaned_df)
    print(f"Removed {removed_count} duplicate rows")
    
    return cleaned_df

def clean_numeric_column(df, column_name):
    """
    Clean a numeric column by converting to float and handling errors.
    
    Args:
        df: pandas DataFrame
        column_name: name of the column to clean
    
    Returns:
        DataFrame with cleaned column
    """
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame")
    
    original_dtype = df[column_name].dtype
    df[column_name] = pd.to_numeric(df[column_name], errors='coerce')
    
    conversion_count = df[column_name].isna().sum() - df[column_name].isna().sum()
    print(f"Converted {conversion_count} values in column '{column_name}'")
    print(f"Original dtype: {original_dtype}, New dtype: {df[column_name].dtype}")
    
    return df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Args:
        df: pandas DataFrame to validate
        required_columns: list of column names that must be present
    
    Returns:
        tuple: (is_valid, error_message)
    """
    if df.empty:
        return False, "DataFrame is empty"
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
    
    if df.isnull().all().any():
        return False, "Some columns contain only null values"
    
    return True, "DataFrame validation passed"

def main():
    """Example usage of data cleaning functions."""
    sample_data = {
        'id': [1, 2, 2, 3, 4, 4, 5],
        'name': ['Alice', 'Bob', 'Bob', 'Charlie', 'David', 'David', 'Eve'],
        'score': ['95', '88', '88', '92', 'invalid', '85', '90']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print()
    
    df_no_dupes = remove_duplicates(df, subset=['id', 'name'])
    print("DataFrame after removing duplicates:")
    print(df_no_dupes)
    print()
    
    df_clean = clean_numeric_column(df_no_dupes, 'score')
    print("DataFrame after cleaning numeric column:")
    print(df_clean)
    print()
    
    is_valid, message = validate_dataframe(df_clean, required_columns=['id', 'name', 'score'])
    print(f"Validation result: {is_valid}")
    print(f"Validation message: {message}")

if __name__ == "__main__":
    main()