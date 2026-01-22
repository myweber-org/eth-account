
import pandas as pd
import numpy as np
from scipy import stats

def remove_missing_rows(df, columns=None):
    if columns is None:
        columns = df.columns
    return df.dropna(subset=columns)

def fill_missing_with_mean(df, columns=None):
    df_copy = df.copy()
    if columns is None:
        columns = df_copy.select_dtypes(include=[np.number]).columns
    for col in columns:
        if col in df_copy.columns:
            df_copy[col] = df_copy[col].fillna(df_copy[col].mean())
    return df_copy

def remove_outliers_iqr(df, columns=None, threshold=1.5):
    df_copy = df.copy()
    if columns is None:
        columns = df_copy.select_dtypes(include=[np.number]).columns
    for col in columns:
        if col in df_copy.columns:
            Q1 = df_copy[col].quantile(0.25)
            Q3 = df_copy[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            df_copy = df_copy[(df_copy[col] >= lower_bound) & (df_copy[col] <= upper_bound)]
    return df_copy

def standardize_columns(df, columns=None):
    df_copy = df.copy()
    if columns is None:
        columns = df_copy.select_dtypes(include=[np.number]).columns
    for col in columns:
        if col in df_copy.columns:
            mean_val = df_copy[col].mean()
            std_val = df_copy[col].std()
            if std_val > 0:
                df_copy[col] = (df_copy[col] - mean_val) / std_val
    return df_copy

def get_data_summary(df):
    summary = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': df.isnull().sum().sum(),
        'numeric_columns': list(df.select_dtypes(include=[np.number]).columns),
        'categorical_columns': list(df.select_dtypes(include=['object']).columns)
    }
    return summary