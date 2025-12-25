
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(dataframe, column, threshold=1.5):
    """
    Remove outliers using the Interquartile Range method.
    """
    Q1 = dataframe[column].quantile(0.25)
    Q3 = dataframe[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR
    return dataframe[(dataframe[column] >= lower_bound) & (dataframe[column] <= upper_bound)]

def normalize_minmax(dataframe, columns):
    """
    Normalize specified columns using Min-Max scaling.
    """
    df_normalized = dataframe.copy()
    for col in columns:
        min_val = df_normalized[col].min()
        max_val = df_normalized[col].max()
        if max_val - min_val != 0:
            df_normalized[col] = (df_normalized[col] - min_val) / (max_val - min_val)
    return df_normalized

def standardize_zscore(dataframe, columns):
    """
    Standardize specified columns using Z-score normalization.
    """
    df_standardized = dataframe.copy()
    for col in columns:
        mean_val = df_standardized[col].mean()
        std_val = df_standardized[col].std()
        if std_val != 0:
            df_standardized[col] = (df_standardized[col] - mean_val) / std_val
    return df_standardized

def handle_missing_values(dataframe, strategy='mean', columns=None):
    """
    Handle missing values using specified strategy.
    """
    df_handled = dataframe.copy()
    if columns is None:
        columns = df_handled.columns
    
    for col in columns:
        if df_handled[col].isnull().any():
            if strategy == 'mean':
                fill_value = df_handled[col].mean()
            elif strategy == 'median':
                fill_value = df_handled[col].median()
            elif strategy == 'mode':
                fill_value = df_handled[col].mode()[0]
            elif strategy == 'drop':
                df_handled = df_handled.dropna(subset=[col])
                continue
            else:
                fill_value = 0
            df_handled[col].fillna(fill_value, inplace=True)
    return df_handled

def clean_dataset(dataframe, numerical_columns, outlier_threshold=1.5, 
                  normalize=True, standardize=False, missing_strategy='mean'):
    """
    Complete data cleaning pipeline.
    """
    df_clean = dataframe.copy()
    
    df_clean = handle_missing_values(df_clean, strategy=missing_strategy, columns=numerical_columns)
    
    for col in numerical_columns:
        df_clean = remove_outliers_iqr(df_clean, col, threshold=outlier_threshold)
    
    if normalize:
        df_clean = normalize_minmax(df_clean, numerical_columns)
    
    if standardize:
        df_clean = standardize_zscore(df_clean, numerical_columns)
    
    return df_clean