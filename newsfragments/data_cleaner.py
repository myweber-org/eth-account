
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(dataframe, column, threshold=1.5):
    """
    Remove outliers from specified column using IQR method.
    """
    if column not in dataframe.columns:
        raise ValueError(f"Column '{column}' not found in dataframe")
    
    q1 = dataframe[column].quantile(0.25)
    q3 = dataframe[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr
    
    filtered_df = dataframe[(dataframe[column] >= lower_bound) & 
                           (dataframe[column] <= upper_bound)]
    return filtered_df

def normalize_minmax(dataframe, columns=None):
    """
    Normalize specified columns using min-max scaling.
    """
    if columns is None:
        columns = dataframe.select_dtypes(include=[np.number]).columns
    
    normalized_df = dataframe.copy()
    for col in columns:
        if col in dataframe.columns and np.issubdtype(dataframe[col].dtype, np.number):
            col_min = dataframe[col].min()
            col_max = dataframe[col].max()
            if col_max != col_min:
                normalized_df[col] = (dataframe[col] - col_min) / (col_max - col_min)
    
    return normalized_df

def standardize_zscore(dataframe, columns=None):
    """
    Standardize specified columns using z-score normalization.
    """
    if columns is None:
        columns = dataframe.select_dtypes(include=[np.number]).columns
    
    standardized_df = dataframe.copy()
    for col in columns:
        if col in dataframe.columns and np.issubdtype(dataframe[col].dtype, np.number):
            mean_val = dataframe[col].mean()
            std_val = dataframe[col].std()
            if std_val > 0:
                standardized_df[col] = (dataframe[col] - mean_val) / std_val
    
    return standardized_df

def handle_missing_values(dataframe, strategy='mean', columns=None):
    """
    Handle missing values in specified columns.
    """
    if columns is None:
        columns = dataframe.select_dtypes(include=[np.number]).columns
    
    filled_df = dataframe.copy()
    
    for col in columns:
        if col not in filled_df.columns:
            continue
            
        missing_mask = filled_df[col].isnull()
        if not missing_mask.any():
            continue
            
        if strategy == 'mean':
            fill_value = filled_df[col].mean()
        elif strategy == 'median':
            fill_value = filled_df[col].median()
        elif strategy == 'mode':
            fill_value = filled_df[col].mode()[0]
        elif strategy == 'zero':
            fill_value = 0
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        filled_df.loc[missing_mask, col] = fill_value
    
    return filled_df

def create_data_quality_report(dataframe):
    """
    Generate a data quality report for the dataframe.
    """
    report = {
        'total_rows': len(dataframe),
        'total_columns': len(dataframe.columns),
        'missing_values': dataframe.isnull().sum().to_dict(),
        'data_types': dataframe.dtypes.astype(str).to_dict(),
        'numeric_stats': {}
    }
    
    numeric_cols = dataframe.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        report['numeric_stats'][col] = {
            'mean': dataframe[col].mean(),
            'std': dataframe[col].std(),
            'min': dataframe[col].min(),
            'max': dataframe[col].max(),
            'median': dataframe[col].median()
        }
    
    return report