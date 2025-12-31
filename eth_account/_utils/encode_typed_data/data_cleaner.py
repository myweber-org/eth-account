import pandas as pd
import numpy as np

def clean_dataset(df, column_mapping=None, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by handling missing values, duplicates, and standardizing columns.
    """
    cleaned_df = df.copy()
    
    if column_mapping:
        cleaned_df = cleaned_df.rename(columns=column_mapping)
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    for column in cleaned_df.select_dtypes(include=[np.number]).columns:
        if cleaned_df[column].isnull().any():
            if fill_missing == 'mean':
                cleaned_df[column].fillna(cleaned_df[column].mean(), inplace=True)
            elif fill_missing == 'median':
                cleaned_df[column].fillna(cleaned_df[column].median(), inplace=True)
            elif fill_missing == 'zero':
                cleaned_df[column].fillna(0, inplace=True)
    
    for column in cleaned_df.select_dtypes(include=['object']).columns:
        cleaned_df[column] = cleaned_df[column].str.strip().str.lower()
        cleaned_df[column].fillna('unknown', inplace=True)
    
    cleaned_df = cleaned_df.reset_index(drop=True)
    
    return cleaned_df

def validate_data(df, required_columns=None, unique_constraints=None):
    """
    Validate the cleaned dataset for required columns and uniqueness constraints.
    """
    validation_results = {}
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        validation_results['missing_columns'] = missing_columns
    
    if unique_constraints:
        duplicate_counts = {}
        for column in unique_constraints:
            if column in df.columns:
                duplicates = df[column].duplicated().sum()
                duplicate_counts[column] = duplicates
        validation_results['duplicate_counts'] = duplicate_counts
    
    validation_results['total_rows'] = len(df)
    validation_results['total_columns'] = len(df.columns)
    validation_results['data_types'] = df.dtypes.to_dict()
    
    return validation_results

if __name__ == "__main__":
    sample_data = {
        'ID': [1, 2, 2, 3, 4, None],
        'Name': ['Alice', 'Bob', 'bob', 'Charlie', 'David', 'Eve'],
        'Age': [25, 30, 30, None, 35, 28],
        'Score': [85.5, 92.0, 92.0, 78.5, None, 88.0]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original Dataset:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    cleaned_df = clean_dataset(df, column_mapping=None, drop_duplicates=True, fill_missing='mean')
    print("Cleaned Dataset:")
    print(cleaned_df)
    print("\n" + "="*50 + "\n")
    
    validation = validate_data(cleaned_df, required_columns=['ID', 'Name', 'Age'], unique_constraints=['ID'])
    print("Validation Results:")
    for key, value in validation.items():
        print(f"{key}: {value}")
import pandas as pd

def clean_dataset(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    drop_duplicates (bool): Whether to drop duplicate rows. Default is True.
    fill_missing (str): Method to fill missing values. Options: 'mean', 'median', 'mode', or 'drop'. Default is 'mean'.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
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

def validate_dataset(df, required_columns=None):
    """
    Validate a DataFrame for required columns and data types.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of required column names. Default is None.
    
    Returns:
    bool: True if validation passes, False otherwise.
    """
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Missing required columns: {missing_columns}")
            return False
    
    if df.empty:
        print("DataFrame is empty")
        return False
    
    return True

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, 2, 3, None],
        'B': [4, None, 6, 6, 8],
        'C': [7, 8, 9, 9, 10]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    
    cleaned = clean_dataset(df, drop_duplicates=True, fill_missing='mean')
    print("\nCleaned DataFrame:")
    print(cleaned)
    
    is_valid = validate_dataset(cleaned, required_columns=['A', 'B', 'C'])
    print(f"\nDataset validation: {is_valid}")