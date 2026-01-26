
import pandas as pd
import numpy as np

def clean_dataset(df, columns_to_check=None, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by handling duplicates and missing values.
    
    Args:
        df: pandas DataFrame to clean
        columns_to_check: list of columns to check for duplicates (default: all columns)
        drop_duplicates: boolean indicating whether to drop duplicate rows
        fill_missing: strategy to fill missing values ('mean', 'median', 'mode', or 'drop')
    
    Returns:
        Cleaned pandas DataFrame
    """
    original_shape = df.shape
    
    # Create a copy to avoid modifying the original
    cleaned_df = df.copy()
    
    # Handle duplicates
    if drop_duplicates:
        if columns_to_check:
            cleaned_df = cleaned_df.drop_duplicates(subset=columns_to_check)
        else:
            cleaned_df = cleaned_df.drop_duplicates()
    
    # Handle missing values
    if fill_missing != 'drop':
        for column in cleaned_df.select_dtypes(include=[np.number]).columns:
            if cleaned_df[column].isnull().any():
                if fill_missing == 'mean':
                    cleaned_df[column].fillna(cleaned_df[column].mean(), inplace=True)
                elif fill_missing == 'median':
                    cleaned_df[column].fillna(cleaned_df[column].median(), inplace=True)
                elif fill_missing == 'mode':
                    cleaned_df[column].fillna(cleaned_df[column].mode()[0], inplace=True)
    else:
        cleaned_df = cleaned_df.dropna()
    
    # Log cleaning results
    print(f"Original dataset shape: {original_shape}")
    print(f"Cleaned dataset shape: {cleaned_df.shape}")
    print(f"Rows removed: {original_shape[0] - cleaned_df.shape[0]}")
    print(f"Columns: {original_shape[1]}")
    
    return cleaned_df

def validate_data(df, required_columns=None, numeric_columns=None):
    """
    Validate the cleaned dataset for basic data quality checks.
    
    Args:
        df: pandas DataFrame to validate
        required_columns: list of columns that must be present
        numeric_columns: list of columns that should be numeric
    
    Returns:
        Dictionary with validation results
    """
    validation_results = {
        'has_duplicates': df.duplicated().any(),
        'has_missing_values': df.isnull().any().any(),
        'row_count': len(df),
        'column_count': len(df.columns)
    }
    
    # Check required columns
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        validation_results['missing_required_columns'] = missing_columns
    
    # Check numeric columns
    if numeric_columns:
        non_numeric_cols = []
        for col in numeric_columns:
            if col in df.columns and not np.issubdtype(df[col].dtype, np.number):
                non_numeric_cols.append(col)
        validation_results['non_numeric_columns'] = non_numeric_cols
    
    return validation_results

# Example usage (commented out for production)
if __name__ == "__main__":
    # Create sample data
    sample_data = {
        'id': [1, 2, 3, 3, 4, 5],
        'value': [10, 20, np.nan, 30, 40, 50],
        'category': ['A', 'B', 'C', 'C', 'A', 'B']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n")
    
    # Clean the data
    cleaned_df = clean_dataset(df, columns_to_check=['id'], fill_missing='mean')
    
    print("\nCleaned DataFrame:")
    print(cleaned_df)
    
    # Validate the cleaned data
    validation = validate_data(cleaned_df, required_columns=['id', 'value'], numeric_columns=['id', 'value'])
    print("\nValidation Results:")
    for key, value in validation.items():
        print(f"{key}: {value}")