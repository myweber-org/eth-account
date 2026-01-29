import pandas as pd

def clean_dataframe(df, remove_duplicates=True, fillna_method=None):
    """
    Clean a pandas DataFrame by handling missing values and optionally removing duplicates.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    remove_duplicates (bool): If True, remove duplicate rows.
    fillna_method (str or None): Method to fill missing values. 
                                 Options: 'ffill', 'bfill', 'mean', 'median', or None to drop rows with nulls.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    # Handle missing values
    if fillna_method is None:
        cleaned_df = cleaned_df.dropna()
    elif fillna_method in ['ffill', 'bfill']:
        cleaned_df = cleaned_df.fillna(method=fillna_method)
    elif fillna_method == 'mean':
        cleaned_df = cleaned_df.fillna(cleaned_df.mean(numeric_only=True))
    elif fillna_method == 'median':
        cleaned_df = cleaned_df.fillna(cleaned_df.median(numeric_only=True))
    else:
        raise ValueError(f"Unsupported fillna_method: {fillna_method}")
    
    # Remove duplicates if requested
    if remove_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    # Reset index after cleaning
    cleaned_df = cleaned_df.reset_index(drop=True)
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of column names that must be present.
    
    Returns:
    tuple: (is_valid, error_message)
    """
    if not isinstance(df, pd.DataFrame):
        return False, "Input is not a pandas DataFrame"
    
    if df.empty:
        return False, "DataFrame is empty"
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
    
    return True, "DataFrame is valid"

# Example usage
if __name__ == "__main__":
    # Create sample data
    sample_data = {
        'A': [1, 2, None, 4, 4],
        'B': [5, None, 7, 8, 8],
        'C': [9, 10, 11, None, 11]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nCleaned DataFrame (fill with forward fill):")
    cleaned = clean_dataframe(df, fillna_method='ffill')
    print(cleaned)
    
    # Validate the cleaned DataFrame
    is_valid, message = validate_dataframe(cleaned, required_columns=['A', 'B', 'C'])
    print(f"\nValidation: {is_valid}, Message: {message}")