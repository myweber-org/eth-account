import pandas as pd
import numpy as np

def remove_duplicates(df, subset=None):
    """
    Remove duplicate rows from DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        subset (list, optional): Columns to consider for duplicates
    
    Returns:
        pd.DataFrame: DataFrame with duplicates removed
    """
    return df.drop_duplicates(subset=subset, keep='first')

def convert_column_types(df, column_types):
    """
    Convert specified columns to given data types.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        column_types (dict): Dictionary mapping columns to target types
    
    Returns:
        pd.DataFrame: DataFrame with converted columns
    """
    result_df = df.copy()
    for column, dtype in column_types.items():
        if column in result_df.columns:
            result_df[column] = result_df[column].astype(dtype)
    return result_df

def handle_missing_values(df, strategy='drop', fill_value=None):
    """
    Handle missing values in DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        strategy (str): 'drop' or 'fill'
        fill_value: Value to fill missing values with
    
    Returns:
        pd.DataFrame: DataFrame with handled missing values
    """
    if strategy == 'drop':
        return df.dropna()
    elif strategy == 'fill':
        return df.fillna(fill_value)
    else:
        raise ValueError("Strategy must be 'drop' or 'fill'")

def normalize_column(df, column):
    """
    Normalize a column to range [0, 1].
    
    Args:
        df (pd.DataFrame): Input DataFrame
        column (str): Column name to normalize
    
    Returns:
        pd.DataFrame: DataFrame with normalized column
    """
    if column not in df.columns:
        return df
    
    result_df = df.copy()
    col_min = result_df[column].min()
    col_max = result_df[column].max()
    
    if col_max != col_min:
        result_df[column] = (result_df[column] - col_min) / (col_max - col_min)
    
    return result_df

def clean_dataframe(df, operations):
    """
    Apply multiple cleaning operations to DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        operations (list): List of cleaning operation configurations
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    result_df = df.copy()
    
    for operation in operations:
        op_type = operation.get('type')
        
        if op_type == 'remove_duplicates':
            result_df = remove_duplicates(result_df, operation.get('subset'))
        
        elif op_type == 'convert_types':
            result_df = convert_column_types(result_df, operation.get('column_types', {}))
        
        elif op_type == 'handle_missing':
            result_df = handle_missing_values(
                result_df,
                strategy=operation.get('strategy', 'drop'),
                fill_value=operation.get('fill_value')
            )
        
        elif op_type == 'normalize':
            result_df = normalize_column(result_df, operation.get('column'))
    
    return result_df

def validate_dataframe(df, checks):
    """
    Validate DataFrame against specified checks.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        checks (dict): Validation checks
    
    Returns:
        dict: Validation results
    """
    results = {}
    
    if 'required_columns' in checks:
        required = set(checks['required_columns'])
        present = set(df.columns)
        results['missing_columns'] = list(required - present)
        results['has_required_columns'] = len(results['missing_columns']) == 0
    
    if 'no_duplicates' in checks and checks['no_duplicates']:
        duplicate_count = df.duplicated().sum()
        results['has_duplicates'] = duplicate_count > 0
        results['duplicate_count'] = duplicate_count
    
    if 'no_null_values' in checks:
        columns = checks['no_null_values']
        if columns == 'all':
            columns = df.columns
        
        null_counts = df[columns].isnull().sum()
        results['null_counts'] = null_counts.to_dict()
        results['has_nulls'] = null_counts.sum() > 0
    
    return results