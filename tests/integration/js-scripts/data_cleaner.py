import pandas as pd
import numpy as np

def remove_duplicates(df):
    """Remove duplicate rows from DataFrame."""
    return df.drop_duplicates()

def fill_missing_values(df, strategy='mean'):
    """Fill missing values using specified strategy."""
    if strategy == 'mean':
        return df.fillna(df.mean())
    elif strategy == 'median':
        return df.fillna(df.median())
    elif strategy == 'mode':
        return df.fillna(df.mode().iloc[0])
    else:
        return df.fillna(0)

def normalize_column(df, column_name):
    """Normalize specified column to range [0,1]."""
    if column_name in df.columns:
        col = df[column_name]
        df[column_name] = (col - col.min()) / (col.max() - col.min())
    return df

def detect_outliers(df, column_name, threshold=3):
    """Detect outliers using z-score method."""
    if column_name in df.columns:
        z_scores = np.abs((df[column_name] - df[column_name].mean()) / df[column_name].std())
        return df[z_scores > threshold]
    return pd.DataFrame()

def clean_dataframe(df, remove_dups=True, fill_na=True, norm_cols=None):
    """Apply multiple cleaning operations to DataFrame."""
    if remove_dups:
        df = remove_duplicates(df)
    
    if fill_na:
        df = fill_missing_values(df)
    
    if norm_cols:
        for col in norm_cols:
            if col in df.columns:
                df = normalize_column(df, col)
    
    return df