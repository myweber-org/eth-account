
import numpy as np
import pandas as pd

def remove_outliers_iqr(data, column, threshold=1.5):
    """
    Remove outliers using IQR method
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    removed_count = len(data) - len(filtered_data)
    
    return filtered_data, removed_count

def normalize_minmax(data, column):
    """
    Normalize column using min-max scaling
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    min_val = data[column].min()
    max_val = data[column].max()
    
    if max_val == min_val:
        return data[column].apply(lambda x: 0.5)
    
    normalized = (data[column] - min_val) / (max_val - min_val)
    return normalized

def standardize_zscore(data, column):
    """
    Standardize column using z-score normalization
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    mean_val = data[column].mean()
    std_val = data[column].std()
    
    if std_val == 0:
        return data[column].apply(lambda x: 0)
    
    standardized = (data[column] - mean_val) / std_val
    return standardized

def handle_missing_values(data, strategy='mean'):
    """
    Handle missing values in numeric columns
    """
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    
    if strategy == 'mean':
        for col in numeric_cols:
            if data[col].isnull().any():
                data[col].fillna(data[col].mean(), inplace=True)
    elif strategy == 'median':
        for col in numeric_cols:
            if data[col].isnull().any():
                data[col].fillna(data[col].median(), inplace=True)
    elif strategy == 'mode':
        for col in numeric_cols:
            if data[col].isnull().any():
                data[col].fillna(data[col].mode()[0], inplace=True)
    elif strategy == 'drop':
        data.dropna(subset=numeric_cols, inplace=True)
    
    return data

def clean_dataset(data, numeric_columns=None, outlier_threshold=1.5, 
                  normalization='standardize', missing_strategy='mean'):
    """
    Comprehensive data cleaning pipeline
    """
    if numeric_columns is None:
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_data = data.copy()
    
    # Handle missing values
    cleaned_data = handle_missing_values(cleaned_data, strategy=missing_strategy)
    
    # Remove outliers and normalize each numeric column
    for column in numeric_columns:
        if column in cleaned_data.columns:
            # Remove outliers
            cleaned_data, removed = remove_outliers_iqr(cleaned_data, column, outlier_threshold)
            
            # Apply normalization
            if normalization == 'minmax':
                cleaned_data[f'{column}_normalized'] = normalize_minmax(cleaned_data, column)
            elif normalization == 'standardize':
                cleaned_data[f'{column}_standardized'] = standardize_zscore(cleaned_data, column)
    
    return cleaned_data

def get_cleaning_summary(original_data, cleaned_data):
    """
    Generate summary of cleaning operations
    """
    summary = {
        'original_rows': len(original_data),
        'cleaned_rows': len(cleaned_data),
        'rows_removed': len(original_data) - len(cleaned_data),
        'removal_percentage': ((len(original_data) - len(cleaned_data)) / len(original_data)) * 100,
        'new_columns_added': [col for col in cleaned_data.columns if col not in original_data.columns]
    }
    
    return summary