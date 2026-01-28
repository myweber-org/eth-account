import pandas as pd

def clean_dataset(df, columns_to_check=None, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by handling duplicates and missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    columns_to_check (list): List of column names to check for duplicates. 
                             If None, checks all columns.
    drop_duplicates (bool): If True, remove duplicate rows.
    fill_missing (str): Strategy to fill missing values. 
                        Options: 'mean', 'median', 'mode', or 'drop'.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        if columns_to_check:
            cleaned_df = cleaned_df.drop_duplicates(subset=columns_to_check)
        else:
            cleaned_df = cleaned_df.drop_duplicates()
    
    if fill_missing == 'drop':
        cleaned_df = cleaned_df.dropna()
    elif fill_missing in ['mean', 'median', 'mode']:
        numeric_cols = cleaned_df.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            if fill_missing == 'mean':
                cleaned_df[col].fillna(cleaned_df[col].mean(), inplace=True)
            elif fill_missing == 'median':
                cleaned_df[col].fillna(cleaned_df[col].median(), inplace=True)
            elif fill_missing == 'mode':
                cleaned_df[col].fillna(cleaned_df[col].mode()[0], inplace=True)
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate a DataFrame for basic integrity checks.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of column names that must be present.
    
    Returns:
    tuple: (is_valid, error_message)
    """
    if df.empty:
        return False, "DataFrame is empty"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    return True, "DataFrame is valid"

if __name__ == "__main__":
    sample_data = {
        'id': [1, 2, 2, 3, 4],
        'value': [10.5, None, 15.0, 20.0, None],
        'category': ['A', 'B', 'B', 'C', 'A']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nCleaned DataFrame (mean imputation):")
    cleaned = clean_dataset(df, columns_to_check=['id'], fill_missing='mean')
    print(cleaned)
    
    is_valid, message = validate_dataframe(cleaned, required_columns=['id', 'value'])
    print(f"\nValidation: {message}")