
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column, factor=1.5):
    """
    Remove outliers using Interquartile Range method.
    Returns filtered data and outlier indices.
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - factor * iqr
    upper_bound = q3 + factor * iqr
    
    mask = (data[column] >= lower_bound) & (data[column] <= upper_bound)
    outliers = data[~mask].index.tolist()
    
    return data[mask].copy(), outliers

def normalize_minmax(data, columns=None):
    """
    Normalize specified columns using Min-Max scaling.
    If columns is None, normalize all numeric columns.
    """
    if columns is None:
        columns = data.select_dtypes(include=[np.number]).columns.tolist()
    
    normalized_data = data.copy()
    for col in columns:
        if col in data.columns and pd.api.types.is_numeric_dtype(data[col]):
            col_min = data[col].min()
            col_max = data[col].max()
            if col_max != col_min:
                normalized_data[col] = (data[col] - col_min) / (col_max - col_min)
            else:
                normalized_data[col] = 0
    
    return normalized_data

def standardize_zscore(data, columns=None):
    """
    Standardize specified columns using Z-score normalization.
    If columns is None, standardize all numeric columns.
    """
    if columns is None:
        columns = data.select_dtypes(include=[np.number]).columns.tolist()
    
    standardized_data = data.copy()
    for col in columns:
        if col in data.columns and pd.api.types.is_numeric_dtype(data[col]):
            col_mean = data[col].mean()
            col_std = data[col].std()
            if col_std > 0:
                standardized_data[col] = (data[col] - col_mean) / col_std
            else:
                standardized_data[col] = 0
    
    return standardized_data

def handle_missing_values(data, strategy='mean', columns=None):
    """
    Handle missing values using specified strategy.
    Supported strategies: 'mean', 'median', 'mode', 'drop'
    """
    if columns is None:
        columns = data.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_data = data.copy()
    
    if strategy == 'drop':
        return cleaned_data.dropna(subset=columns)
    
    for col in columns:
        if col in cleaned_data.columns and pd.api.types.is_numeric_dtype(cleaned_data[col]):
            if strategy == 'mean':
                fill_value = cleaned_data[col].mean()
            elif strategy == 'median':
                fill_value = cleaned_data[col].median()
            elif strategy == 'mode':
                fill_value = cleaned_data[col].mode()[0] if not cleaned_data[col].mode().empty else 0
            else:
                raise ValueError(f"Unsupported strategy: {strategy}")
            
            cleaned_data[col] = cleaned_data[col].fillna(fill_value)
    
    return cleaned_data

def create_data_summary(data):
    """
    Create a comprehensive summary of the dataset.
    Returns a dictionary with various statistics.
    """
    summary = {
        'shape': data.shape,
        'columns': data.columns.tolist(),
        'dtypes': data.dtypes.to_dict(),
        'missing_values': data.isnull().sum().to_dict(),
        'numeric_stats': {},
        'categorical_stats': {}
    }
    
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        summary['numeric_stats'][col] = {
            'mean': data[col].mean(),
            'median': data[col].median(),
            'std': data[col].std(),
            'min': data[col].min(),
            'max': data[col].max(),
            'skewness': data[col].skew(),
            'kurtosis': data[col].kurtosis()
        }
    
    categorical_cols = data.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        summary['categorical_stats'][col] = {
            'unique_count': data[col].nunique(),
            'top_value': data[col].mode()[0] if not data[col].mode().empty else None,
            'top_count': data[col].value_counts().iloc[0] if not data[col].value_counts().empty else 0
        }
    
    return summary