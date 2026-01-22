def remove_duplicates(input_list):
    seen = set()
    result = []
    for item in input_list:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return resultimport pandas as pd
import numpy as np

def remove_duplicates(df, subset=None):
    """
    Remove duplicate rows from DataFrame.
    """
    return df.drop_duplicates(subset=subset, keep='first')

def fill_missing_values(df, strategy='mean', columns=None):
    """
    Fill missing values in specified columns.
    """
    df_filled = df.copy()
    
    if columns is None:
        columns = df.columns
    
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
    Normalize values in a column.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    df_normalized = df.copy()
    
    if method == 'minmax':
        min_val = df[column].min()
        max_val = df[column].max()
        if max_val != min_val:
            df_normalized[column] = (df[column] - min_val) / (max_val - min_val)
    
    elif method == 'zscore':
        mean_val = df[column].mean()
        std_val = df[column].std()
        if std_val != 0:
            df_normalized[column] = (df[column] - mean_val) / std_val
    
    return df_normalized

def filter_outliers(df, column, method='iqr', threshold=1.5):
    """
    Filter outliers from a column.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    df_filtered = df.copy()
    
    if method == 'iqr':
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        
        mask = (df[column] >= lower_bound) & (df[column] <= upper_bound)
        df_filtered = df[mask].reset_index(drop=True)
    
    return df_filtered

def clean_dataframe(df, operations):
    """
    Apply multiple cleaning operations to DataFrame.
    """
    cleaned_df = df.copy()
    
    for operation in operations:
        if operation['type'] == 'remove_duplicates':
            cleaned_df = remove_duplicates(cleaned_df, **operation.get('params', {}))
        elif operation['type'] == 'fill_missing':
            cleaned_df = fill_missing_values(cleaned_df, **operation.get('params', {}))
        elif operation['type'] == 'normalize':
            cleaned_df = normalize_column(cleaned_df, **operation.get('params', {}))
        elif operation['type'] == 'filter_outliers':
            cleaned_df = filter_outliers(cleaned_df, **operation.get('params', {}))
    
    return cleaned_df