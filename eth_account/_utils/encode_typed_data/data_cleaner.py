
import pandas as pd
import re

def clean_text_column(df, column_name):
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame")
    
    df[column_name] = df[column_name].astype(str).str.strip()
    df[column_name] = df[column_name].str.lower()
    df[column_name] = df[column_name].apply(lambda x: re.sub(r'\s+', ' ', x))
    
    return df

def remove_duplicates(df, subset=None, keep='first'):
    return df.drop_duplicates(subset=subset, keep=keep)

def standardize_dates(df, column_name, date_format='%Y-%m-%d'):
    df[column_name] = pd.to_datetime(df[column_name], errors='coerce')
    df[column_name] = df[column_name].dt.strftime(date_format)
    return df

def clean_dataset(df, text_columns=None, date_columns=None, deduplicate=True):
    if text_columns:
        for col in text_columns:
            df = clean_text_column(df, col)
    
    if date_columns:
        for col in date_columns:
            df = standardize_dates(df, col)
    
    if deduplicate:
        df = remove_duplicates(df)
    
    return df