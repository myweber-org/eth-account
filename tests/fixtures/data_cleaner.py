
import numpy as np
import pandas as pd

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def normalize_minmax(df, column):
    min_val = df[column].min()
    max_val = df[column].max()
    df[column + '_normalized'] = (df[column] - min_val) / (max_val - min_val)
    return df

def standardize_zscore(df, column):
    mean_val = df[column].mean()
    std_val = df[column].std()
    df[column + '_standardized'] = (df[column] - mean_val) / std_val
    return df

def handle_missing_values(df, strategy='mean'):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if strategy == 'mean':
        for col in numeric_cols:
            df[col].fillna(df[col].mean(), inplace=True)
    elif strategy == 'median':
        for col in numeric_cols:
            df[col].fillna(df[col].median(), inplace=True)
    elif strategy == 'mode':
        for col in numeric_cols:
            df[col].fillna(df[col].mode()[0], inplace=True)
    
    return df

def clean_dataset(df, outlier_columns=None, normalize_columns=None, standardize_columns=None, missing_strategy='mean'):
    df_clean = df.copy()
    
    if outlier_columns:
        for col in outlier_columns:
            df_clean = remove_outliers_iqr(df_clean, col)
    
    df_clean = handle_missing_values(df_clean, strategy=missing_strategy)
    
    if normalize_columns:
        for col in normalize_columns:
            df_clean = normalize_minmax(df_clean, col)
    
    if standardize_columns:
        for col in standardize_columns:
            df_clean = standardize_zscore(df_clean, col)
    
    return df_clean