
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(dataframe, columns, factor=1.5):
    """
    Remove outliers using IQR method
    """
    df_clean = dataframe.copy()
    for col in columns:
        if col in df_clean.columns:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - factor * IQR
            upper_bound = Q3 + factor * IQR
            df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
    return df_clean

def remove_outliers_zscore(dataframe, columns, threshold=3):
    """
    Remove outliers using Z-score method
    """
    df_clean = dataframe.copy()
    for col in columns:
        if col in df_clean.columns:
            z_scores = np.abs(stats.zscore(df_clean[col]))
            df_clean = df_clean[z_scores < threshold]
    return df_clean

def normalize_minmax(dataframe, columns):
    """
    Normalize data using Min-Max scaling
    """
    df_normalized = dataframe.copy()
    for col in columns:
        if col in df_normalized.columns:
            min_val = df_normalized[col].min()
            max_val = df_normalized[col].max()
            if max_val != min_val:
                df_normalized[col] = (df_normalized[col] - min_val) / (max_val - min_val)
    return df_normalized

def normalize_zscore(dataframe, columns):
    """
    Normalize data using Z-score standardization
    """
    df_normalized = dataframe.copy()
    for col in columns:
        if col in df_normalized.columns:
            mean_val = df_normalized[col].mean()
            std_val = df_normalized[col].std()
            if std_val > 0:
                df_normalized[col] = (df_normalized[col] - mean_val) / std_val
    return df_normalized

def handle_missing_values(dataframe, strategy='mean', columns=None):
    """
    Handle missing values using specified strategy
    """
    df_processed = dataframe.copy()
    if columns is None:
        columns = df_processed.columns
    
    for col in columns:
        if col in df_processed.columns:
            if strategy == 'mean':
                fill_value = df_processed[col].mean()
            elif strategy == 'median':
                fill_value = df_processed[col].median()
            elif strategy == 'mode':
                fill_value = df_processed[col].mode()[0]
            elif strategy == 'drop':
                df_processed = df_processed.dropna(subset=[col])
                continue
            else:
                fill_value = 0
            
            df_processed[col] = df_processed[col].fillna(fill_value)
    
    return df_processed

def clean_dataset(dataframe, config):
    """
    Main cleaning function with configurable pipeline
    """
    df = dataframe.copy()
    
    if 'missing_values' in config:
        df = handle_missing_values(
            df, 
            strategy=config['missing_values'].get('strategy', 'mean'),
            columns=config['missing_values'].get('columns')
        )
    
    if 'outliers' in config:
        method = config['outliers'].get('method', 'iqr')
        columns = config['outliers'].get('columns', df.columns.tolist())
        
        if method == 'iqr':
            df = remove_outliers_iqr(df, columns, config['outliers'].get('factor', 1.5))
        elif method == 'zscore':
            df = remove_outliers_zscore(df, columns, config['outliers'].get('threshold', 3))
    
    if 'normalization' in config:
        method = config['normalization'].get('method', 'minmax')
        columns = config['normalization'].get('columns', df.columns.tolist())
        
        if method == 'minmax':
            df = normalize_minmax(df, columns)
        elif method == 'zscore':
            df = normalize_zscore(df, columns)
    
    return df

def get_cleaning_report(dataframe):
    """
    Generate cleaning report for dataset
    """
    report = {
        'original_shape': dataframe.shape,
        'missing_values': dataframe.isnull().sum().to_dict(),
        'numeric_columns': dataframe.select_dtypes(include=[np.number]).columns.tolist(),
        'categorical_columns': dataframe.select_dtypes(exclude=[np.number]).columns.tolist()
    }
    
    numeric_cols = dataframe.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        report['descriptive_stats'] = dataframe[numeric_cols].describe().to_dict()
    
    return report