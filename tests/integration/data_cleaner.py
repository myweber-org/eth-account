import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column, factor=1.5):
    """
    Remove outliers using IQR method
    """
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - factor * iqr
    upper_bound = q3 + factor * iqr
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data

def zscore_normalize(data, column):
    """
    Normalize data using z-score method
    """
    mean = data[column].mean()
    std = data[column].std()
    data[column + '_normalized'] = (data[column] - mean) / std
    return data

def minmax_normalize(data, column, feature_range=(0, 1)):
    """
    Normalize data using min-max scaling
    """
    min_val = data[column].min()
    max_val = data[column].max()
    
    if max_val - min_val == 0:
        data[column + '_normalized'] = 0
    else:
        data[column + '_normalized'] = (data[column] - min_val) / (max_val - min_val)
    
    if feature_range != (0, 1):
        new_min, new_max = feature_range
        data[column + '_normalized'] = data[column + '_normalized'] * (new_max - new_min) + new_min
    
    return data

def handle_missing_values(data, strategy='mean'):
    """
    Handle missing values in numerical columns
    """
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        if data[col].isnull().any():
            if strategy == 'mean':
                fill_value = data[col].mean()
            elif strategy == 'median':
                fill_value = data[col].median()
            elif strategy == 'mode':
                fill_value = data[col].mode()[0]
            else:
                fill_value = 0
            
            data[col] = data[col].fillna(fill_value)
    
    return data

def clean_dataset(data, numeric_columns=None, outlier_method='iqr', normalize_method='zscore'):
    """
    Main cleaning pipeline for datasets
    """
    cleaned_data = data.copy()
    
    if numeric_columns is None:
        numeric_columns = cleaned_data.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_data = handle_missing_values(cleaned_data)
    
    for column in numeric_columns:
        if column in cleaned_data.columns:
            if outlier_method == 'iqr':
                cleaned_data = remove_outliers_iqr(cleaned_data, column)
            
            if normalize_method == 'zscore':
                cleaned_data = zscore_normalize(cleaned_data, column)
            elif normalize_method == 'minmax':
                cleaned_data = minmax_normalize(cleaned_data, column)
    
    return cleaned_data

def validate_data(data, check_duplicates=True, check_infinite=True):
    """
    Validate cleaned dataset
    """
    validation_report = {}
    
    if check_duplicates:
        validation_report['duplicates'] = data.duplicated().sum()
    
    if check_infinite:
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        infinite_count = 0
        for col in numeric_cols:
            infinite_count += np.isinf(data[col]).sum()
        validation_report['infinite_values'] = infinite_count
    
    validation_report['missing_values'] = data.isnull().sum().sum()
    validation_report['shape'] = data.shape
    
    return validation_report