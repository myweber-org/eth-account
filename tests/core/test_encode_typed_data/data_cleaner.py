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

def normalize_column(df, column, method='minmax'):
    """
    Normalize a column using specified method.
    """
    if method == 'minmax':
        min_val = df[column].min()
        max_val = df[column].max()
        if max_val != min_val:
            df[column] = (df[column] - min_val) / (max_val - min_val)
    elif method == 'zscore':
        mean_val = df[column].mean()
        std_val = df[column].std()
        if std_val != 0:
            df[column] = (df[column] - mean_val) / std_val
    
    return df

def clean_dataframe(df, operations=None):
    """
    Apply multiple cleaning operations to DataFrame.
    """
    if operations is None:
        operations = []
    
    cleaned_df = df.copy()
    
    for operation in operations:
        if operation['type'] == 'remove_duplicates':
            cleaned_df = remove_duplicates(cleaned_df, subset=operation.get('subset'))
        elif operation['type'] == 'fill_missing':
            cleaned_df = fill_missing_values(
                cleaned_df, 
                strategy=operation.get('strategy', 'mean'),
                columns=operation.get('columns')
            )
        elif operation['type'] == 'normalize':
            cleaned_df = normalize_column(
                cleaned_df,
                column=operation['column'],
                method=operation.get('method', 'minmax')
            )
    
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