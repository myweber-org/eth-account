
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(dataframe, column, threshold=1.5):
    """
    Remove outliers from specified column using IQR method
    """
    if column not in dataframe.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    q1 = dataframe[column].quantile(0.25)
    q3 = dataframe[column].quantile(0.75)
    iqr = q3 - q1
    
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr
    
    filtered_df = dataframe[(dataframe[column] >= lower_bound) & 
                           (dataframe[column] <= upper_bound)]
    
    return filtered_df.copy()

def normalize_column_zscore(dataframe, column):
    """
    Normalize column using z-score normalization
    """
    if column not in dataframe.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    normalized_col = stats.zscore(dataframe[column])
    result_df = dataframe.copy()
    result_df[f"{column}_normalized"] = normalized_col
    
    return result_df

def min_max_normalize(dataframe, column, new_min=0, new_max=1):
    """
    Apply min-max normalization to specified column
    """
    if column not in dataframe.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    old_min = dataframe[column].min()
    old_max = dataframe[column].max()
    
    if old_max == old_min:
        normalized_values = np.zeros(len(dataframe))
    else:
        normalized_values = ((dataframe[column] - old_min) / (old_max - old_min)) * (new_max - new_min) + new_min
    
    result_df = dataframe.copy()
    result_df[f"{column}_scaled"] = normalized_values
    
    return result_df

def handle_missing_values(dataframe, strategy='mean', columns=None):
    """
    Handle missing values in specified columns
    """
    result_df = dataframe.copy()
    
    if columns is None:
        columns = dataframe.columns
    
    for col in columns:
        if col not in dataframe.columns:
            continue
            
        if dataframe[col].isnull().any():
            if strategy == 'mean':
                fill_value = dataframe[col].mean()
            elif strategy == 'median':
                fill_value = dataframe[col].median()
            elif strategy == 'mode':
                fill_value = dataframe[col].mode()[0]
            elif strategy == 'drop':
                result_df = result_df.dropna(subset=[col])
                continue
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
            
            result_df[col] = result_df[col].fillna(fill_value)
    
    return result_df

def create_cleaning_pipeline(dataframe, operations):
    """
    Apply multiple cleaning operations in sequence
    """
    result_df = dataframe.copy()
    
    for operation in operations:
        if operation['type'] == 'remove_outliers':
            result_df = remove_outliers_iqr(result_df, operation['column'], 
                                          operation.get('threshold', 1.5))
        elif operation['type'] == 'normalize':
            result_df = normalize_column_zscore(result_df, operation['column'])
        elif operation['type'] == 'scale':
            result_df = min_max_normalize(result_df, operation['column'],
                                        operation.get('new_min', 0),
                                        operation.get('new_max', 1))
        elif operation['type'] == 'handle_missing':
            result_df = handle_missing_values(result_df,
                                            operation.get('strategy', 'mean'),
                                            operation.get('columns'))
    
    return result_df