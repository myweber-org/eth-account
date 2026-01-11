import pandas as pd

def clean_dataframe(df, drop_duplicates=True, fill_missing=False, fill_value=0):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df: pandas DataFrame to clean.
        drop_duplicates: If True, remove duplicate rows.
        fill_missing: If True, fill missing values with fill_value.
        fill_value: Value to use for filling missing data.
    
    Returns:
        Cleaned pandas DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    if fill_missing:
        cleaned_df = cleaned_df.fillna(fill_value)
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Args:
        df: pandas DataFrame to validate.
        required_columns: List of column names that must be present.
    
    Returns:
        Tuple of (is_valid, error_message).
    """
    if df.empty:
        return False, "DataFrame is empty"
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
    
    return True, "DataFrame is valid"
import pandas as pd
import numpy as np

def clean_dataset(df, drop_duplicates=True, fill_missing=True, fill_strategy='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean
    drop_duplicates (bool): Whether to remove duplicate rows
    fill_missing (bool): Whether to fill missing values
    fill_strategy (str): Strategy for filling missing values ('mean', 'median', 'mode', or 'zero')
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    original_shape = df.shape
    
    # Create a copy to avoid modifying the original
    cleaned_df = df.copy()
    
    # Remove duplicate rows
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
        duplicates_removed = original_shape[0] - cleaned_df.shape[0]
        print(f"Removed {duplicates_removed} duplicate rows")
    
    # Handle missing values
    if fill_missing:
        for column in cleaned_df.columns:
            if cleaned_df[column].isnull().any():
                missing_count = cleaned_df[column].isnull().sum()
                
                if cleaned_df[column].dtype in ['int64', 'float64']:
                    if fill_strategy == 'mean':
                        fill_value = cleaned_df[column].mean()
                    elif fill_strategy == 'median':
                        fill_value = cleaned_df[column].median()
                    elif fill_strategy == 'mode':
                        fill_value = cleaned_df[column].mode()[0]
                    elif fill_strategy == 'zero':
                        fill_value = 0
                    else:
                        fill_value = cleaned_df[column].mean()
                    
                    cleaned_df[column].fillna(fill_value, inplace=True)
                    print(f"Filled {missing_count} missing values in column '{column}' with {fill_strategy}: {fill_value}")
                
                elif cleaned_df[column].dtype == 'object':
                    # For categorical columns, fill with mode
                    fill_value = cleaned_df[column].mode()[0] if not cleaned_df[column].mode().empty else 'Unknown'
                    cleaned_df[column].fillna(fill_value, inplace=True)
                    print(f"Filled {missing_count} missing values in column '{column}' with mode: '{fill_value}'")
    
    # Report cleaning summary
    print(f"\nCleaning Summary:")
    print(f"Original shape: {original_shape}")
    print(f"Cleaned shape: {cleaned_df.shape}")
    print(f"Rows removed: {original_shape[0] - cleaned_df.shape[0]}")
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and data types.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate
    required_columns (list): List of required column names
    
    Returns:
    dict: Validation results
    """
    validation_results = {
        'is_valid': True,
        'errors': [],
        'warnings': []
    }
    
    # Check if DataFrame is empty
    if df.empty:
        validation_results['is_valid'] = False
        validation_results['errors'].append("DataFrame is empty")
    
    # Check required columns
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            validation_results['is_valid'] = False
            validation_results['errors'].append(f"Missing required columns: {missing_columns}")
    
    # Check for infinite values in numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if np.isinf(df[col]).any():
            validation_results['warnings'].append(f"Column '{col}' contains infinite values")
    
    # Check for negative values in columns that shouldn't have them
    non_negative_cols = [col for col in numeric_cols if 'age' in col.lower() or 'count' in col.lower()]
    for col in non_negative_cols:
        if (df[col] < 0).any():
            validation_results['warnings'].append(f"Column '{col}' contains negative values")
    
    return validation_results

# Example usage
if __name__ == "__main__":
    # Create sample data with duplicates and missing values
    sample_data = {
        'id': [1, 2, 3, 4, 5, 1, 2],
        'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve', 'Alice', 'Bob'],
        'age': [25, 30, np.nan, 35, 40, 25, 30],
        'score': [85.5, 92.0, 78.5, np.nan, 88.0, 85.5, 92.0],
        'department': ['HR', 'IT', 'IT', 'Finance', 'HR', 'HR', 'IT']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    # Clean the dataset
    cleaned_df = clean_dataset(df, drop_duplicates=True, fill_missing=True, fill_strategy='mean')
    
    print("\n" + "="*50 + "\n")
    print("Cleaned DataFrame:")
    print(cleaned_df)
    
    # Validate the cleaned dataset
    print("\n" + "="*50 + "\n")
    validation = validate_dataframe(cleaned_df, required_columns=['id', 'name', 'age', 'score'])
    print("Validation Results:")
    print(f"Is valid: {validation['is_valid']}")
    if validation['errors']:
        print("Errors:", validation['errors'])
    if validation['warnings']:
        print("Warnings:", validation['warnings'])