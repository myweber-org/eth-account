import numpy as np
import pandas as pd

def remove_outliers_iqr(df, columns):
    """
    Remove outliers using IQR method for specified columns.
    Returns cleaned DataFrame and outlier indices.
    """
    outlier_indices = []
    cleaned_df = df.copy()
    
    for col in columns:
        if col not in cleaned_df.columns:
            continue
            
        Q1 = cleaned_df[col].quantile(0.25)
        Q3 = cleaned_df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        col_outliers = cleaned_df[(cleaned_df[col] < lower_bound) | (cleaned_df[col] > upper_bound)].index
        outlier_indices.extend(col_outliers)
    
    outlier_indices = list(set(outlier_indices))
    cleaned_df = cleaned_df.drop(outlier_indices)
    
    return cleaned_df, outlier_indices

def normalize_minmax(df, columns):
    """
    Apply min-max normalization to specified columns.
    Returns DataFrame with normalized values.
    """
    normalized_df = df.copy()
    
    for col in columns:
        if col not in normalized_df.columns:
            continue
            
        col_min = normalized_df[col].min()
        col_max = normalized_df[col].max()
        
        if col_max - col_min > 0:
            normalized_df[col] = (normalized_df[col] - col_min) / (col_max - col_min)
        else:
            normalized_df[col] = 0
    
    return normalized_df

def standardize_zscore(df, columns):
    """
    Apply z-score standardization to specified columns.
    Returns DataFrame with standardized values.
    """
    standardized_df = df.copy()
    
    for col in columns:
        if col not in standardized_df.columns:
            continue
            
        col_mean = standardized_df[col].mean()
        col_std = standardized_df[col].std()
        
        if col_std > 0:
            standardized_df[col] = (standardized_df[col] - col_mean) / col_std
        else:
            standardized_df[col] = 0
    
    return standardized_df

def handle_missing_values(df, strategy='mean', columns=None):
    """
    Handle missing values using specified strategy.
    Supported strategies: 'mean', 'median', 'mode', 'drop'
    """
    processed_df = df.copy()
    
    if columns is None:
        columns = processed_df.columns
    
    for col in columns:
        if col not in processed_df.columns:
            continue
            
        if strategy == 'drop':
            processed_df = processed_df.dropna(subset=[col])
        elif strategy == 'mean':
            processed_df[col] = processed_df[col].fillna(processed_df[col].mean())
        elif strategy == 'median':
            processed_df[col] = processed_df[col].fillna(processed_df[col].median())
        elif strategy == 'mode':
            processed_df[col] = processed_df[col].fillna(processed_df[col].mode()[0])
    
    return processed_df

def validate_dataframe(df, required_columns=None, min_rows=1):
    """
    Validate DataFrame structure and content.
    Returns boolean and error message if validation fails.
    """
    if not isinstance(df, pd.DataFrame):
        return False, "Input is not a pandas DataFrame"
    
    if len(df) < min_rows:
        return False, f"DataFrame has fewer than {min_rows} rows"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    return True, "Validation passed"