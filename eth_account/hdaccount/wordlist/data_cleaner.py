import pandas as pd
import numpy as np

def clean_dataset(df, drop_duplicates=True, convert_types=True):
    """
    Clean a pandas DataFrame by removing duplicates and converting data types.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean
    drop_duplicates (bool): Whether to remove duplicate rows
    convert_types (bool): Whether to convert columns to optimal data types
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        removed = initial_rows - len(cleaned_df)
        print(f"Removed {removed} duplicate rows")
    
    if convert_types:
        for col in cleaned_df.columns:
            if cleaned_df[col].dtype == 'object':
                try:
                    cleaned_df[col] = pd.to_datetime(cleaned_df[col])
                    print(f"Converted column '{col}' to datetime")
                except (ValueError, TypeError):
                    try:
                        cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='ignore')
                    except:
                        pass
    
    cleaned_df = cleaned_df.reset_index(drop=True)
    
    missing_values = cleaned_df.isnull().sum().sum()
    if missing_values > 0:
        print(f"Warning: Dataset contains {missing_values} missing values")
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate
    required_columns (list): List of column names that must be present
    
    Returns:
    dict: Validation results
    """
    validation_results = {
        'is_valid': True,
        'errors': [],
        'warnings': [],
        'summary': {}
    }
    
    if not isinstance(df, pd.DataFrame):
        validation_results['is_valid'] = False
        validation_results['errors'].append("Input is not a pandas DataFrame")
        return validation_results
    
    validation_results['summary']['total_rows'] = len(df)
    validation_results['summary']['total_columns'] = len(df.columns)
    validation_results['summary']['column_names'] = list(df.columns)
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            validation_results['is_valid'] = False
            validation_results['errors'].append(f"Missing required columns: {missing_columns}")
    
    if df.empty:
        validation_results['warnings'].append("DataFrame is empty")
    
    duplicate_rows = df.duplicated().sum()
    if duplicate_rows > 0:
        validation_results['warnings'].append(f"Found {duplicate_rows} duplicate rows")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        validation_results['summary']['numeric_columns'] = list(numeric_cols)
    
    return validation_results

def sample_data_processing():
    """Example usage of the data cleaning functions."""
    data = {
        'id': [1, 2, 2, 3, 4, 4],
        'name': ['Alice', 'Bob', 'Bob', 'Charlie', 'David', 'David'],
        'score': ['85', '92', '92', '78', '95', '95'],
        'date': ['2023-01-01', '2023-01-02', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-04']
    }
    
    df = pd.DataFrame(data)
    print("Original DataFrame:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    cleaned = clean_dataset(df)
    print("Cleaned DataFrame:")
    print(cleaned)
    print("\n" + "="*50 + "\n")
    
    validation = validate_dataframe(cleaned, required_columns=['id', 'name'])
    print("Validation Results:")
    for key, value in validation.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    sample_data_processing()