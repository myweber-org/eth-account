import pandas as pd

def clean_dataset(df, drop_duplicates=True, fill_missing=False, fill_value=0):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean
        drop_duplicates (bool): Whether to remove duplicate rows
        fill_missing (bool): Whether to fill missing values
        fill_value: Value to use for filling missing data
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        removed = initial_rows - len(cleaned_df)
        print(f"Removed {removed} duplicate rows")
    
    if fill_missing:
        missing_before = cleaned_df.isnull().sum().sum()
        cleaned_df = cleaned_df.fillna(fill_value)
        missing_after = cleaned_df.isnull().sum().sum()
        print(f"Filled {missing_before - missing_after} missing values")
    
    return cleaned_df

def validate_dataset(df, required_columns=None):
    """
    Validate dataset structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        required_columns (list): List of required column names
    
    Returns:
        dict: Validation results
    """
    validation_results = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': df.isnull().sum().sum(),
        'duplicate_rows': df.duplicated().sum()
    }
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        validation_results['missing_required_columns'] = missing_columns
        validation_results['all_required_columns_present'] = len(missing_columns) == 0
    
    return validation_results

def sample_data_cleaning():
    """Example usage of the data cleaning functions."""
    data = {
        'id': [1, 2, 2, 3, 4],
        'name': ['Alice', 'Bob', 'Bob', None, 'Eve'],
        'score': [85, 92, 92, 78, None]
    }
    
    df = pd.DataFrame(data)
    print("Original dataset:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    cleaned = clean_dataset(df, drop_duplicates=True, fill_missing=True, fill_value=0)
    print("Cleaned dataset:")
    print(cleaned)
    print("\n" + "="*50 + "\n")
    
    validation = validate_dataset(cleaned, required_columns=['id', 'name', 'score'])
    print("Validation results:")
    for key, value in validation.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    sample_data_cleaning()import pandas as pd

def clean_dataset(df, drop_duplicates=True, fill_na_method='drop'):
    """
    Clean a pandas DataFrame by handling missing values and duplicates.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean
        drop_duplicates (bool): Whether to drop duplicate rows
        fill_na_method (str): Method to handle NaN values ('drop', 'mean', 'median', 'mode', 'zero')
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    # Handle missing values
    if fill_na_method == 'drop':
        cleaned_df = cleaned_df.dropna()
    elif fill_na_method == 'mean':
        cleaned_df = cleaned_df.fillna(cleaned_df.mean(numeric_only=True))
    elif fill_na_method == 'median':
        cleaned_df = cleaned_df.fillna(cleaned_df.median(numeric_only=True))
    elif fill_na_method == 'mode':
        cleaned_df = cleaned_df.fillna(cleaned_df.mode().iloc[0])
    elif fill_na_method == 'zero':
        cleaned_df = cleaned_df.fillna(0)
    
    # Remove duplicates if requested
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    # Reset index after cleaning
    cleaned_df = cleaned_df.reset_index(drop=True)
    
    return cleaned_df

def validate_dataset(df, required_columns=None):
    """
    Validate dataset structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        required_columns (list): List of required column names
    
    Returns:
        dict: Validation results
    """
    validation_results = {
        'is_valid': True,
        'errors': [],
        'warnings': [],
        'stats': {}
    }
    
    # Check if DataFrame is empty
    if df.empty:
        validation_results['is_valid'] = False
        validation_results['errors'].append('DataFrame is empty')
    
    # Check required columns
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            validation_results['is_valid'] = False
            validation_results['errors'].append(f'Missing required columns: {missing_columns}')
    
    # Calculate basic statistics
    validation_results['stats']['row_count'] = len(df)
    validation_results['stats']['column_count'] = len(df.columns)
    validation_results['stats']['null_count'] = df.isnull().sum().sum()
    validation_results['stats']['duplicate_count'] = df.duplicated().sum()
    
    # Add warnings for potential issues
    if validation_results['stats']['null_count'] > 0:
        validation_results['warnings'].append(f'Dataset contains {validation_results["stats"]["null_count"]} null values')
    
    if validation_results['stats']['duplicate_count'] > 0:
        validation_results['warnings'].append(f'Dataset contains {validation_results["stats"]["duplicate_count"]} duplicate rows')
    
    return validation_results

# Example usage (commented out for production)
# if __name__ == "__main__":
#     # Create sample data
#     sample_data = {
#         'A': [1, 2, None, 4, 5, 5],
#         'B': [10, 20, 30, None, 50, 50],
#         'C': ['x', 'y', 'z', 'x', 'y', 'y']
#     }
#     
#     df = pd.DataFrame(sample_data)
#     print("Original DataFrame:")
#     print(df)
#     
#     # Validate dataset
#     validation = validate_dataset(df, required_columns=['A', 'B'])
#     print("\nValidation Results:")
#     print(validation)
#     
#     # Clean dataset
#     cleaned = clean_dataset(df, drop_duplicates=True, fill_na_method='mean')
#     print("\nCleaned DataFrame:")
#     print(cleaned)