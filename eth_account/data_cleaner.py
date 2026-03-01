
import pandas as pd
import numpy as np

def clean_dataset(df, deduplicate=True, handle_nulls='drop'):
    """
    Clean a pandas DataFrame by removing duplicates and handling null values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    deduplicate (bool): If True, remove duplicate rows.
    handle_nulls (str): Strategy for null values - 'drop', 'fill_mean', or 'fill_median'.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if deduplicate:
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        removed = initial_rows - len(cleaned_df)
        print(f"Removed {removed} duplicate rows.")
    
    if handle_nulls == 'drop':
        cleaned_df = cleaned_df.dropna()
        print("Dropped rows with null values.")
    elif handle_nulls == 'fill_mean':
        numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
        cleaned_df[numeric_cols] = cleaned_df[numeric_cols].fillna(cleaned_df[numeric_cols].mean())
        print("Filled numeric nulls with column means.")
    elif handle_nulls == 'fill_median':
        numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
        cleaned_df[numeric_cols] = cleaned_df[numeric_cols].fillna(cleaned_df[numeric_cols].median())
        print("Filled numeric nulls with column medians.")
    
    return cleaned_df

def validate_dataset(df, required_columns=None):
    """
    Validate a DataFrame for basic integrity checks.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of column names that must be present.
    
    Returns:
    dict: Dictionary with validation results.
    """
    validation_results = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'null_count': df.isnull().sum().sum(),
        'duplicate_rows': df.duplicated().sum()
    }
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        validation_results['missing_required_columns'] = missing_columns
        validation_results['all_required_present'] = len(missing_columns) == 0
    
    return validation_results

if __name__ == "__main__":
    sample_data = {
        'id': [1, 2, 2, 3, 4],
        'value': [10.5, None, 20.3, 15.7, None],
        'category': ['A', 'B', 'B', 'A', 'C']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original dataset:")
    print(df)
    print("\nValidation results:")
    print(validate_dataset(df))
    
    cleaned = clean_dataset(df, deduplicate=True, handle_nulls='fill_mean')
    print("\nCleaned dataset:")
    print(cleaned)
    print("\nCleaned validation results:")
    print(validate_dataset(cleaned))