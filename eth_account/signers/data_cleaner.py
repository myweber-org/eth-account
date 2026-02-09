
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column, factor=1.5):
    """
    Remove outliers using the Interquartile Range method
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - factor * IQR
    upper_bound = Q3 + factor * IQR
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    removed_count = len(data) - len(filtered_data)
    
    return filtered_data, removed_count

def remove_outliers_zscore(data, column, threshold=3):
    """
    Remove outliers using Z-score method
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    z_scores = np.abs(stats.zscore(data[column].dropna()))
    filtered_indices = np.where(z_scores < threshold)[0]
    
    filtered_data = data.iloc[filtered_indices].copy()
    removed_count = len(data) - len(filtered_data)
    
    return filtered_data, removed_count

def normalize_minmax(data, column):
    """
    Normalize data using Min-Max scaling
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    min_val = data[column].min()
    max_val = data[column].max()
    
    if max_val == min_val:
        return data[column].copy()
    
    normalized = (data[column] - min_val) / (max_val - min_val)
    return normalized

def normalize_zscore(data, column):
    """
    Normalize data using Z-score standardization
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    mean_val = data[column].mean()
    std_val = data[column].std()
    
    if std_val == 0:
        return data[column].copy()
    
    standardized = (data[column] - mean_val) / std_val
    return standardized

def clean_dataset(data, numeric_columns=None, outlier_method='iqr', normalize_method='minmax'):
    """
    Comprehensive data cleaning pipeline
    """
    if numeric_columns is None:
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_data = data.copy()
    cleaning_report = {}
    
    for column in numeric_columns:
        if column not in cleaned_data.columns:
            continue
            
        original_count = len(cleaned_data)
        
        if outlier_method == 'iqr':
            cleaned_data, removed = remove_outliers_iqr(cleaned_data, column)
        elif outlier_method == 'zscore':
            cleaned_data, removed = remove_outliers_zscore(cleaned_data, column)
        else:
            removed = 0
        
        if normalize_method == 'minmax':
            cleaned_data[f'{column}_normalized'] = normalize_minmax(cleaned_data, column)
        elif normalize_method == 'zscore':
            cleaned_data[f'{column}_normalized'] = normalize_zscore(cleaned_data, column)
        
        cleaning_report[column] = {
            'original_samples': original_count,
            'outliers_removed': removed,
            'remaining_samples': len(cleaned_data),
            'normalization_applied': normalize_method
        }
    
    return cleaned_data, cleaning_report

def validate_data(data, required_columns=None, allow_nan_ratio=0.1):
    """
    Validate dataset structure and quality
    """
    validation_results = {
        'is_valid': True,
        'issues': [],
        'missing_values': {}
    }
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in data.columns]
        if missing_cols:
            validation_results['is_valid'] = False
            validation_results['issues'].append(f"Missing required columns: {missing_cols}")
    
    for column in data.columns:
        nan_count = data[column].isna().sum()
        nan_ratio = nan_count / len(data)
        
        validation_results['missing_values'][column] = {
            'count': int(nan_count),
            'ratio': float(nan_ratio)
        }
        
        if nan_ratio > allow_nan_ratio:
            validation_results['is_valid'] = False
            validation_results['issues'].append(
                f"Column '{column}' has {nan_ratio:.1%} missing values (threshold: {allow_nan_ratio:.1%})"
            )
    
    validation_results['total_rows'] = len(data)
    validation_results['total_columns'] = len(data.columns)
    
    return validation_results