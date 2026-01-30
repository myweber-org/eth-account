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

def z_score_normalization(data, column):
    """
    Apply z-score normalization to specified column
    """
    mean = data[column].mean()
    std = data[column].std()
    
    if std > 0:
        data[f'{column}_normalized'] = (data[column] - mean) / std
    else:
        data[f'{column}_normalized'] = 0
    
    return data

def min_max_normalization(data, column, new_min=0, new_max=1):
    """
    Apply min-max normalization to specified column
    """
    old_min = data[column].min()
    old_max = data[column].max()
    
    if old_max - old_min > 0:
        data[f'{column}_scaled'] = ((data[column] - old_min) / (old_max - old_min)) * (new_max - new_min) + new_min
    else:
        data[f'{column}_scaled'] = new_min
    
    return data

def clean_dataset(df, numeric_columns):
    """
    Main cleaning function that processes multiple numeric columns
    """
    cleaned_df = df.copy()
    
    for col in numeric_columns:
        if col in cleaned_df.columns:
            # Remove outliers
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
            
            # Apply both normalizations
            cleaned_df = z_score_normalization(cleaned_df, col)
            cleaned_df = min_max_normalization(cleaned_df, col)
    
    return cleaned_df

def validate_data(df, required_columns):
    """
    Validate that required columns exist and have no null values
    """
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    null_counts = df[required_columns].isnull().sum()
    columns_with_nulls = null_counts[null_counts > 0].index.tolist()
    
    return {
        'is_valid': len(missing_columns) == 0 and len(columns_with_nulls) == 0,
        'missing_columns': missing_columns,
        'columns_with_nulls': columns_with_nulls
    }