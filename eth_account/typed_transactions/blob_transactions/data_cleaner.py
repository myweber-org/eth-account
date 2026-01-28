import pandas as pd
import numpy as np

def remove_missing_values(df, threshold=0.5):
    """
    Remove columns with missing values exceeding threshold percentage.
    """
    missing_percent = df.isnull().sum() / len(df)
    columns_to_drop = missing_percent[missing_percent > threshold].index
    return df.drop(columns=columns_to_drop)

def normalize_numeric_columns(df, columns=None):
    """
    Normalize specified numeric columns using min-max scaling.
    If columns is None, normalize all numeric columns.
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    for col in columns:
        if col in df.columns:
            min_val = df[col].min()
            max_val = df[col].max()
            if max_val > min_val:
                df[col] = (df[col] - min_val) / (max_val - min_val)
    return df

def encode_categorical(df, columns=None, method='onehot'):
    """
    Encode categorical columns using specified method.
    Supported methods: 'onehot', 'label'
    """
    if columns is None:
        columns = df.select_dtypes(include=['object', 'category']).columns
    
    if method == 'onehot':
        return pd.get_dummies(df, columns=columns, drop_first=True)
    elif method == 'label':
        for col in columns:
            if col in df.columns:
                df[col] = df[col].astype('category').cat.codes
        return df
    else:
        raise ValueError("Method must be 'onehot' or 'label'")

def clean_dataset(df, missing_threshold=0.5, normalize=True, encode=True):
    """
    Complete data cleaning pipeline.
    """
    df_clean = df.copy()
    df_clean = remove_missing_values(df_clean, missing_threshold)
    
    if normalize:
        df_clean = normalize_numeric_columns(df_clean)
    
    if encode:
        df_clean = encode_categorical(df_clean)
    
    return df_clean