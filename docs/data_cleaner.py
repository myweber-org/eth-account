
import pandas as pd

def clean_dataset(df, columns_to_check=None, fill_na_method='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df: pandas DataFrame to clean
        columns_to_check: list of columns to check for duplicates (default: all columns)
        fill_na_method: method to fill missing values ('mean', 'median', 'mode', or 'drop')
    
    Returns:
        Cleaned pandas DataFrame
    """
    # Create a copy to avoid modifying the original
    cleaned_df = df.copy()
    
    # Remove duplicates
    if columns_to_check:
        cleaned_df = cleaned_df.drop_duplicates(subset=columns_to_check)
    else:
        cleaned_df = cleaned_df.drop_duplicates()
    
    # Handle missing values
    if fill_na_method == 'drop':
        cleaned_df = cleaned_df.dropna()
    elif fill_na_method in ['mean', 'median']:
        numeric_cols = cleaned_df.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            if fill_na_method == 'mean':
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mean())
            else:
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
    elif fill_na_method == 'mode':
        for col in cleaned_df.columns:
            if cleaned_df[col].dtype == 'object':
                mode_value = cleaned_df[col].mode()
                if not mode_value.empty:
                    cleaned_df[col] = cleaned_df[col].fillna(mode_value[0])
    
    # Reset index after cleaning
    cleaned_df = cleaned_df.reset_index(drop=True)
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate that a DataFrame meets basic requirements.
    
    Args:
        df: pandas DataFrame to validate
        required_columns: list of columns that must be present
    
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
import pandas as pd
import numpy as np

def clean_dataset(df, text_columns=None, fill_strategy='mean'):
    """
    Clean a pandas DataFrame by handling missing values and standardizing text.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean
    text_columns (list): List of column names containing text data
    fill_strategy (str): Strategy for filling numeric missing values ('mean', 'median', 'mode')
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    
    df_clean = df.copy()
    
    # Handle missing values in numeric columns
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        if df_clean[col].isnull().any():
            if fill_strategy == 'mean':
                fill_value = df_clean[col].mean()
            elif fill_strategy == 'median':
                fill_value = df_clean[col].median()
            elif fill_strategy == 'mode':
                fill_value = df_clean[col].mode()[0]
            else:
                fill_value = 0
            df_clean[col].fillna(fill_value, inplace=True)
    
    # Standardize text columns
    if text_columns:
        for col in text_columns:
            if col in df_clean.columns:
                # Convert to string, strip whitespace, and convert to lowercase
                df_clean[col] = df_clean[col].astype(str).str.strip().str.lower()
    
    # Remove duplicate rows
    df_clean = df_clean.drop_duplicates()
    
    # Reset index after cleaning
    df_clean.reset_index(drop=True, inplace=True)
    
    return df_clean

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate
    required_columns (list): List of required column names
    
    Returns:
    tuple: (is_valid, error_message)
    """
    
    if not isinstance(df, pd.DataFrame):
        return False, "Input is not a pandas DataFrame"
    
    if df.empty:
        return False, "DataFrame is empty"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    return True, "DataFrame is valid"

# Example usage (commented out for production)
# if __name__ == "__main__":
#     # Create sample data
#     sample_data = {
#         'id': [1, 2, 3, 4, 5],
#         'name': ['Alice', 'Bob', None, 'David', 'Eve'],
#         'age': [25, 30, 35, None, 28],
#         'score': [85.5, 92.0, 78.5, 88.0, 95.5]
#     }
#     
#     df = pd.DataFrame(sample_data)
#     print("Original DataFrame:")
#     print(df)
#     print("\nCleaned DataFrame:")
#     cleaned_df = clean_dataset(df, text_columns=['name'], fill_strategy='mean')
#     print(cleaned_df)
#     
#     is_valid, message = validate_dataframe(cleaned_df, required_columns=['id', 'name', 'age'])
#     print(f"\nValidation: {is_valid} - {message}")
import pandas as pd
import numpy as np

def clean_dataset(df, column_mapping=None, drop_duplicates=True, fill_missing=True):
    """
    Clean a pandas DataFrame by handling duplicates and missing values.
    
    Args:
        df: pandas DataFrame to clean
        column_mapping: dictionary for renaming columns
        drop_duplicates: boolean to remove duplicate rows
        fill_missing: boolean to fill missing values
    
    Returns:
        Cleaned pandas DataFrame
    """
    cleaned_df = df.copy()
    
    if column_mapping:
        cleaned_df = cleaned_df.rename(columns=column_mapping)
    
    if drop_duplicates:
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        removed = initial_rows - len(cleaned_df)
        print(f"Removed {removed} duplicate rows")
    
    if fill_missing:
        for col in cleaned_df.columns:
            if cleaned_df[col].dtype in ['int64', 'float64']:
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
            elif cleaned_df[col].dtype == 'object':
                cleaned_df[col] = cleaned_df[col].fillna('Unknown')
    
    cleaned_df = cleaned_df.reset_index(drop=True)
    
    print(f"Dataset cleaned: {len(cleaned_df)} rows, {len(cleaned_df.columns)} columns")
    return cleaned_df

def validate_data(df, required_columns=None, numeric_columns=None):
    """
    Validate the structure and content of a DataFrame.
    
    Args:
        df: pandas DataFrame to validate
        required_columns: list of columns that must be present
        numeric_columns: list of columns that should be numeric
    
    Returns:
        Dictionary with validation results
    """
    validation_results = {
        'has_data': not df.empty,
        'row_count': len(df),
        'column_count': len(df.columns),
        'missing_values': df.isnull().sum().to_dict(),
        'duplicates': df.duplicated().sum()
    }
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        validation_results['missing_required_columns'] = missing_cols
    
    if numeric_columns:
        non_numeric = []
        for col in numeric_columns:
            if col in df.columns and not np.issubdtype(df[col].dtype, np.number):
                non_numeric.append(col)
        validation_results['non_numeric_columns'] = non_numeric
    
    return validation_results

def sample_data_cleaning():
    """
    Example usage of the data cleaning functions.
    """
    np.random.seed(42)
    
    sample_data = {
        'id': list(range(1, 11)) + [5, 6],
        'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve', 'Frank', 
                'Grace', 'Henry', 'Ivy', 'Jack', 'Bob', 'Charlie'],
        'age': [25, 30, 35, np.nan, 28, 32, 40, 45, 29, 31, 30, 35],
        'score': [85.5, 92.0, 78.5, 88.0, np.nan, 95.5, 82.0, 79.5, 91.0, 87.5, 92.0, 78.5],
        'department': ['Sales', 'IT', 'HR', 'IT', 'Sales', 'IT', 'HR', 'IT', 'Sales', 'IT', 'IT', 'HR']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original dataset:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    cleaned = clean_dataset(df, drop_duplicates=True, fill_missing=True)
    print("\nCleaned dataset:")
    print(cleaned)
    
    validation = validate_data(cleaned, 
                              required_columns=['id', 'name', 'age', 'score'],
                              numeric_columns=['age', 'score'])
    
    print("\nValidation results:")
    for key, value in validation.items():
        print(f"{key}: {value}")
    
    return cleaned

if __name__ == "__main__":
    cleaned_data = sample_data_cleaning()