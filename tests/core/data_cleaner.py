
def remove_duplicates_preserve_order(iterable):
    seen = set()
    result = []
    for item in iterable:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result
import pandas as pd

def clean_dataset(df, columns_to_check=None, fill_na_method='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    columns_to_check (list, optional): List of column names to check for duplicates.
                                       If None, checks all columns.
    fill_na_method (str): Method to fill missing values ('mean', 'median', 'mode', or 'drop').
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    original_shape = df.shape
    
    # Remove duplicates
    if columns_to_check is None:
        df_cleaned = df.drop_duplicates()
    else:
        df_cleaned = df.drop_duplicates(subset=columns_to_check)
    
    # Handle missing values
    if fill_na_method == 'drop':
        df_cleaned = df_cleaned.dropna()
    elif fill_na_method in ['mean', 'median']:
        numeric_cols = df_cleaned.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            if fill_na_method == 'mean':
                fill_value = df_cleaned[col].mean()
            else:
                fill_value = df_cleaned[col].median()
            df_cleaned[col] = df_cleaned[col].fillna(fill_value)
    elif fill_na_method == 'mode':
        for col in df_cleaned.columns:
            mode_value = df_cleaned[col].mode()
            if not mode_value.empty:
                df_cleaned[col] = df_cleaned[col].fillna(mode_value.iloc[0])
    
    # Print cleaning summary
    print(f"Original dataset shape: {original_shape}")
    print(f"Cleaned dataset shape: {df_cleaned.shape}")
    print(f"Rows removed: {original_shape[0] - df_cleaned.shape[0]}")
    print(f"Columns: {original_shape[1]}")
    
    return df_cleaned

def validate_dataframe(df, required_columns=None, min_rows=1):
    """
    Validate that a DataFrame meets basic requirements.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list, optional): List of required column names.
    min_rows (int): Minimum number of rows required.
    
    Returns:
    bool: True if validation passes, False otherwise.
    """
    if not isinstance(df, pd.DataFrame):
        print("Error: Input is not a pandas DataFrame")
        return False
    
    if df.empty:
        print("Error: DataFrame is empty")
        return False
    
    if len(df) < min_rows:
        print(f"Error: DataFrame has fewer than {min_rows} rows")
        return False
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            print(f"Error: Missing required columns: {missing_cols}")
            return False
    
    return True

# Example usage
if __name__ == "__main__":
    # Create sample data
    data = {
        'id': [1, 2, 2, 3, 4, 5],
        'name': ['Alice', 'Bob', 'Bob', 'Charlie', None, 'Eve'],
        'age': [25, 30, 30, None, 35, 28],
        'score': [85.5, 92.0, 92.0, 78.5, None, 88.0]
    }
    
    df = pd.DataFrame(data)
    print("Original DataFrame:")
    print(df)
    print("\n")
    
    # Clean the data
    cleaned_df = clean_dataset(df, columns_to_check=['id', 'name'], fill_na_method='mean')
    
    print("\nCleaned DataFrame:")
    print(cleaned_df)
    
    # Validate the cleaned data
    is_valid = validate_dataframe(cleaned_df, required_columns=['id', 'name', 'age', 'score'], min_rows=1)
    print(f"\nData validation passed: {is_valid}")