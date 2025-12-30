import pandas as pd

def clean_dataframe(df, drop_duplicates=True, fill_missing=None):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    drop_duplicates (bool): Whether to drop duplicate rows.
    fill_missing (str or dict): Method to fill missing values.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    if fill_missing is not None:
        if isinstance(fill_missing, dict):
            cleaned_df = cleaned_df.fillna(fill_missing)
        else:
            cleaned_df = cleaned_df.fillna(fill_missing)
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and required columns.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of required column names.
    
    Returns:
    bool: True if validation passes, False otherwise.
    """
    if not isinstance(df, pd.DataFrame):
        return False
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Missing required columns: {missing_columns}")
            return False
    
    return True

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, None, 4, 1],
        'B': [5, None, 7, 8, 5],
        'C': ['x', 'y', 'z', 'x', 'x']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    
    cleaned = clean_dataframe(df, fill_missing=0)
    print("\nCleaned DataFrame:")
    print(cleaned)
    
    is_valid = validate_dataframe(cleaned, required_columns=['A', 'B', 'C'])
    print(f"\nDataFrame validation: {is_valid}")import pandas as pd
import numpy as np
from pathlib import Path

def clean_csv_data(input_path, output_path=None, strategy='mean'):
    """
    Clean CSV data by handling missing values and standardizing numeric columns.
    
    Parameters:
    input_path (str): Path to input CSV file
    output_path (str, optional): Path for cleaned output CSV. If None, returns DataFrame
    strategy (str): Strategy for missing values - 'mean', 'median', 'drop', or 'zero'
    
    Returns:
    pd.DataFrame or None: Cleaned DataFrame if output_path is None, else None
    """
    
    if not Path(input_path).exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    df = pd.read_csv(input_path)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        if df[col].isnull().any():
            if strategy == 'mean':
                fill_value = df[col].mean()
            elif strategy == 'median':
                fill_value = df[col].median()
            elif strategy == 'drop':
                df = df.dropna(subset=[col])
                continue
            elif strategy == 'zero':
                fill_value = 0
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
            
            df[col] = df[col].fillna(fill_value)
    
    non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns
    for col in non_numeric_cols:
        df[col] = df[col].fillna('Unknown')
    
    df = df.reset_index(drop=True)
    
    if output_path:
        df.to_csv(output_path, index=False)
        return None
    else:
        return df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate
    required_columns (list, optional): List of required column names
    
    Returns:
    dict: Validation results with status and messages
    """
    
    validation_result = {
        'is_valid': True,
        'messages': [],
        'missing_columns': []
    }
    
    if required_columns:
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            validation_result['is_valid'] = False
            validation_result['missing_columns'] = missing
            validation_result['messages'].append(f"Missing required columns: {missing}")
    
    if df.empty:
        validation_result['is_valid'] = False
        validation_result['messages'].append("DataFrame is empty")
    
    duplicate_rows = df.duplicated().sum()
    if duplicate_rows > 0:
        validation_result['messages'].append(f"Found {duplicate_rows} duplicate rows")
    
    return validation_result

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'id': [1, 2, 3, 4, 5],
        'value': [10.5, np.nan, 15.2, np.nan, 20.1],
        'category': ['A', 'B', None, 'A', 'C'],
        'score': [85, 92, np.nan, 78, 88]
    })
    
    temp_input = "temp_sample.csv"
    temp_output = "temp_cleaned.csv"
    
    sample_data.to_csv(temp_input, index=False)
    
    cleaned_df = clean_csv_data(temp_input, strategy='mean')
    
    validation = validate_dataframe(cleaned_df, required_columns=['id', 'value', 'category', 'score'])
    
    print(f"Validation status: {validation['is_valid']}")
    print(f"Messages: {validation['messages']}")
    
    import os
    os.remove(temp_input)
    
    if os.path.exists(temp_output):
        os.remove(temp_output)