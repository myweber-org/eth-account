import pandas as pd
import numpy as np

def remove_duplicates(df, subset=None):
    """
    Remove duplicate rows from DataFrame.
    """
    return df.drop_duplicates(subset=subset, keep='first')

def fill_missing_values(df, strategy='mean', columns=None):
    """
    Fill missing values using specified strategy.
    """
    if columns is None:
        columns = df.columns
    
    df_filled = df.copy()
    
    for col in columns:
        if df[col].dtype in [np.float64, np.int64]:
            if strategy == 'mean':
                fill_value = df[col].mean()
            elif strategy == 'median':
                fill_value = df[col].median()
            elif strategy == 'mode':
                fill_value = df[col].mode()[0]
            else:
                fill_value = 0
            
            df_filled[col] = df[col].fillna(fill_value)
        else:
            df_filled[col] = df[col].fillna('Unknown')
    
    return df_filled

def normalize_column(df, column):
    """
    Normalize numeric column to range [0, 1].
    """
    if df[column].dtype in [np.float64, np.int64]:
        min_val = df[column].min()
        max_val = df[column].max()
        
        if max_val > min_val:
            df[column] = (df[column] - min_val) / (max_val - min_val)
    
    return df

def clean_dataframe(df, operations=None):
    """
    Apply multiple cleaning operations to DataFrame.
    """
    if operations is None:
        operations = ['remove_duplicates', 'fill_missing']
    
    cleaned_df = df.copy()
    
    if 'remove_duplicates' in operations:
        cleaned_df = remove_duplicates(cleaned_df)
    
    if 'fill_missing' in operations:
        cleaned_df = fill_missing_values(cleaned_df)
    
    return cleaned_df

def validate_dataframe(df, rules=None):
    """
    Validate DataFrame against specified rules.
    """
    if rules is None:
        rules = {}
    
    validation_results = {}
    
    for column, rule in rules.items():
        if column in df.columns:
            if 'min' in rule:
                validation_results[f'{column}_min'] = df[column].min() >= rule['min']
            if 'max' in rule:
                validation_results[f'{column}_max'] = df[column].max() <= rule['max']
            if 'not_null' in rule and rule['not_null']:
                validation_results[f'{column}_not_null'] = df[column].isnull().sum() == 0
    
    return validation_results