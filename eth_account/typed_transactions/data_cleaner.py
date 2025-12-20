import pandas as pd

def clean_dataset(df, subset_columns=None, fill_strategy='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        subset_columns (list, optional): Columns to consider for duplicate removal.
        fill_strategy (str): Strategy for filling missing values ('mean', 'median', 'mode', or 'drop').
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    # Remove duplicates
    if subset_columns:
        cleaned_df = cleaned_df.drop_duplicates(subset=subset_columns)
    else:
        cleaned_df = cleaned_df.drop_duplicates()
    
    # Handle missing values
    if fill_strategy == 'drop':
        cleaned_df = cleaned_df.dropna()
    elif fill_strategy in ['mean', 'median']:
        numeric_cols = cleaned_df.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            if fill_strategy == 'mean':
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mean())
            elif fill_strategy == 'median':
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
    elif fill_strategy == 'mode':
        for col in cleaned_df.columns:
            mode_val = cleaned_df[col].mode()
            if not mode_val.empty:
                cleaned_df[col] = cleaned_df[col].fillna(mode_val.iloc[0])
    
    return cleaned_df

def validate_data(df, required_columns=None, min_rows=1):
    """
    Validate DataFrame structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate.
        required_columns (list): List of required column names.
        min_rows (int): Minimum number of rows required.
    
    Returns:
        tuple: (is_valid, error_message)
    """
    if df.empty:
        return False, "DataFrame is empty"
    
    if len(df) < min_rows:
        return False, f"DataFrame has fewer than {min_rows} rows"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    return True, "Data validation passed"

# Example usage
if __name__ == "__main__":
    # Create sample data
    sample_data = {
        'id': [1, 2, 2, 3, 4],
        'value': [10.5, None, 15.0, 20.0, None],
        'category': ['A', 'B', 'B', 'A', 'C']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    
    # Clean the data
    cleaned = clean_dataset(df, subset_columns=['id'], fill_strategy='mean')
    print("\nCleaned DataFrame:")
    print(cleaned)
    
    # Validate the cleaned data
    is_valid, message = validate_data(cleaned, required_columns=['id', 'value'], min_rows=2)
    print(f"\nValidation: {is_valid} - {message}")