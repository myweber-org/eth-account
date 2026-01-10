
import pandas as pd

def clean_dataset(df, remove_duplicates=True, fill_method='drop'):
    """
    Clean a pandas DataFrame by handling missing values and optionally removing duplicates.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean
    remove_duplicates (bool): If True, remove duplicate rows
    fill_method (str): Method for handling nulls - 'drop', 'mean', 'median', or 'zero'
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    original_shape = df.shape
    
    # Handle missing values
    if fill_method == 'drop':
        df_cleaned = df.dropna()
    elif fill_method == 'mean':
        df_cleaned = df.fillna(df.mean(numeric_only=True))
    elif fill_method == 'median':
        df_cleaned = df.fillna(df.median(numeric_only=True))
    elif fill_method == 'zero':
        df_cleaned = df.fillna(0)
    else:
        df_cleaned = df.copy()
    
    # Remove duplicates if requested
    if remove_duplicates:
        df_cleaned = df_cleaned.drop_duplicates()
    
    # Report cleaning statistics
    rows_removed = original_shape[0] - df_cleaned.shape[0]
    cols_removed = original_shape[1] - df_cleaned.shape[1]
    
    print(f"Original shape: {original_shape}")
    print(f"Cleaned shape: {df_cleaned.shape}")
    print(f"Rows removed: {rows_removed}")
    print(f"Columns removed: {cols_removed}")
    
    return df_cleaned

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate
    required_columns (list): List of column names that must be present
    
    Returns:
    bool: True if validation passes, False otherwise
    """
    if not isinstance(df, pd.DataFrame):
        print("Error: Input is not a pandas DataFrame")
        return False
    
    if df.empty:
        print("Warning: DataFrame is empty")
        return True
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            print(f"Error: Missing required columns: {missing_cols}")
            return False
    
    return True

# Example usage
if __name__ == "__main__":
    # Create sample data with missing values and duplicates
    sample_data = {
        'id': [1, 2, 3, 3, 4, 5],
        'value': [10.5, None, 15.2, 15.2, None, 20.1],
        'category': ['A', 'B', 'C', 'C', 'A', 'B']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n")
    
    # Clean the data
    cleaned_df = clean_dataset(df, remove_duplicates=True, fill_method='mean')
    print("\nCleaned DataFrame:")
    print(cleaned_df)