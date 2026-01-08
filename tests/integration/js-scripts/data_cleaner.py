import pandas as pd

def remove_duplicates(dataframe, subset=None, keep='first'):
    """
    Remove duplicate rows from a pandas DataFrame.
    
    Args:
        dataframe: Input DataFrame
        subset: Column label or sequence of labels to consider for duplicates
        keep: 'first', 'last', or False to mark all duplicates
    
    Returns:
        DataFrame with duplicates removed
    """
    if subset is None:
        subset = dataframe.columns.tolist()
    
    cleaned_df = dataframe.drop_duplicates(subset=subset, keep=keep)
    
    print(f"Removed {len(dataframe) - len(cleaned_df)} duplicate rows")
    print(f"Original shape: {dataframe.shape}, Cleaned shape: {cleaned_df.shape}")
    
    return cleaned_df

def validate_dataframe(dataframe, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Args:
        dataframe: DataFrame to validate
        required_columns: List of columns that must be present
    
    Returns:
        Boolean indicating if validation passed
    """
    if dataframe.empty:
        print("Warning: DataFrame is empty")
        return False
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in dataframe.columns]
        if missing_columns:
            print(f"Missing required columns: {missing_columns}")
            return False
    
    if dataframe.isnull().all().any():
        print("Warning: Some columns contain only null values")
    
    return True

def clean_numeric_columns(dataframe, columns=None):
    """
    Clean numeric columns by converting to appropriate types.
    
    Args:
        dataframe: Input DataFrame
        columns: List of columns to clean (defaults to all numeric columns)
    
    Returns:
        DataFrame with cleaned numeric columns
    """
    if columns is None:
        columns = dataframe.select_dtypes(include=['number']).columns.tolist()
    
    cleaned_df = dataframe.copy()
    
    for column in columns:
        if column in cleaned_df.columns:
            try:
                cleaned_df[column] = pd.to_numeric(cleaned_df[column], errors='coerce')
            except Exception as e:
                print(f"Error cleaning column {column}: {e}")
    
    return cleaned_df

def main():
    """
    Example usage of data cleaning functions.
    """
    sample_data = {
        'id': [1, 2, 3, 1, 2, 4],
        'name': ['Alice', 'Bob', 'Charlie', 'Alice', 'Bob', 'David'],
        'age': [25, 30, 35, 25, 30, 40],
        'score': [85.5, 92.0, 78.5, 85.5, 92.0, 88.0]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print()
    
    cleaned_df = remove_duplicates(df, subset=['id', 'name'])
    print()
    
    is_valid = validate_dataframe(cleaned_df, required_columns=['id', 'name', 'age'])
    print(f"DataFrame validation: {is_valid}")
    
    cleaned_df = clean_numeric_columns(cleaned_df)
    print("\nCleaned DataFrame:")
    print(cleaned_df)

if __name__ == "__main__":
    main()