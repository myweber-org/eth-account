
import numpy as np
import pandas as pd

def remove_outliers_iqr(data, column, multiplier=1.5):
    """
    Remove outliers using IQR method
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data.copy()

def normalize_minmax(data, column):
    """
    Normalize data using min-max scaling
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    min_val = data[column].min()
    max_val = data[column].max()
    
    if max_val == min_val:
        return data[column].copy()
    
    normalized = (data[column] - min_val) / (max_val - min_val)
    return normalized

def standardize_zscore(data, column):
    """
    Standardize data using z-score normalization
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    mean_val = data[column].mean()
    std_val = data[column].std()
    
    if std_val == 0:
        return data[column].copy()
    
    standardized = (data[column] - mean_val) / std_val
    return standardized

def handle_missing_values(data, strategy='mean', columns=None):
    """
    Handle missing values in specified columns
    """
    if columns is None:
        columns = data.columns
    
    data_copy = data.copy()
    
    for column in columns:
        if column not in data.columns:
            continue
            
        if data[column].isnull().any():
            if strategy == 'mean':
                fill_value = data[column].mean()
            elif strategy == 'median':
                fill_value = data[column].median()
            elif strategy == 'mode':
                fill_value = data[column].mode()[0]
            elif strategy == 'constant':
                fill_value = 0
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
            
            data_copy[column] = data[column].fillna(fill_value)
    
    return data_copy

def clean_dataset(data, outlier_columns=None, normalize_columns=None, 
                  standardize_columns=None, missing_strategy='mean'):
    """
    Comprehensive data cleaning pipeline
    """
    cleaned_data = data.copy()
    
    # Handle missing values
    cleaned_data = handle_missing_values(cleaned_data, strategy=missing_strategy)
    
    # Remove outliers
    if outlier_columns:
        for column in outlier_columns:
            if column in cleaned_data.columns:
                cleaned_data = remove_outliers_iqr(cleaned_data, column)
    
    # Normalize specified columns
    if normalize_columns:
        for column in normalize_columns:
            if column in cleaned_data.columns:
                cleaned_data[column] = normalize_minmax(cleaned_data, column)
    
    # Standardize specified columns
    if standardize_columns:
        for column in standardize_columns:
            if column in cleaned_data.columns:
                cleaned_data[column] = standardize_zscore(cleaned_data, column)
    
    return cleaned_data