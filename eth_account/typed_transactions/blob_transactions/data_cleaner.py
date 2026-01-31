import pandas as pd
import re

def clean_dataframe(df, columns_to_clean=None, remove_duplicates=True, case_normalization='lower'):
    """
    Clean a pandas DataFrame by removing duplicates and normalizing string columns.
    """
    cleaned_df = df.copy()
    
    if remove_duplicates:
        initial_rows = len(cleaned_df)
        cleaned_df.drop_duplicates(inplace=True)
        removed = initial_rows - len(cleaned_df)
        print(f"Removed {removed} duplicate rows.")
    
    if columns_to_clean is None:
        columns_to_clean = cleaned_df.select_dtypes(include=['object']).columns
    
    for col in columns_to_clean:
        if col in cleaned_df.columns and cleaned_df[col].dtype == 'object':
            if case_normalization == 'lower':
                cleaned_df[col] = cleaned_df[col].str.lower()
            elif case_normalization == 'upper':
                cleaned_df[col] = cleaned_df[col].str.upper()
            elif case_normalization == 'title':
                cleaned_df[col] = cleaned_df[col].str.title()
            
            cleaned_df[col] = cleaned_df[col].str.strip()
            cleaned_df[col] = cleaned_df[col].replace(r'\s+', ' ', regex=True)
    
    return cleaned_df

def validate_email_column(df, email_column):
    """
    Validate email addresses in a specified column and return a boolean mask.
    """
    if email_column not in df.columns:
        raise ValueError(f"Column '{email_column}' not found in DataFrame.")
    
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return df[email_column].str.match(email_pattern, na=False)

def remove_special_characters(text, keep_chars="a-zA-Z0-9\s"):
    """
    Remove special characters from a string, keeping only specified character sets.
    """
    if pd.isna(text):
        return text
    pattern = f'[^{keep_chars}]'
    return re.sub(pattern, '', str(text))import pandas as pd
import numpy as np

def remove_duplicates(df, subset=None):
    """
    Remove duplicate rows from a DataFrame.
    """
    return df.drop_duplicates(subset=subset, keep='first')

def fill_missing_values(df, strategy='mean', columns=None):
    """
    Fill missing values in specified columns using a given strategy.
    """
    if columns is None:
        columns = df.columns
    
    df_filled = df.copy()
    for col in columns:
        if df[col].dtype in [np.float64, np.int64]:
            if strategy == 'mean':
                fill_value = df[col].mean()
            elif strategy == 'median':
                fill_value = df[col].median()
            elif strategy == 'mode':
                fill_value = df[col].mode()[0]
            else:
                fill_value = 0
            df_filled[col].fillna(fill_value, inplace=True)
        else:
            df_filled[col].fillna('Unknown', inplace=True)
    return df_filled

def normalize_column(df, column):
    """
    Normalize a numeric column to range [0, 1].
    """
    if df[column].dtype in [np.float64, np.int64]:
        min_val = df[column].min()
        max_val = df[column].max()
        if max_val > min_val:
            df[column] = (df[column] - min_val) / (max_val - min_val)
    return df

def clean_dataframe(df, duplicate_subset=None, fill_strategy='mean', normalize_cols=None):
    """
    Perform a complete cleaning pipeline on a DataFrame.
    """
    df_clean = remove_duplicates(df, subset=duplicate_subset)
    df_clean = fill_missing_values(df_clean, strategy=fill_strategy)
    
    if normalize_cols:
        for col in normalize_cols:
            df_clean = normalize_column(df_clean, col)
    
    return df_clean

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, 2, 4, 5],
        'B': [5.0, np.nan, 7.0, 8.0, 9.0],
        'C': ['x', 'y', 'y', 'z', 'x']
    }
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    
    cleaned_df = clean_dataframe(df, duplicate_subset=['A'], fill_strategy='mean', normalize_cols=['A', 'B'])
    print("\nCleaned DataFrame:")
    print(cleaned_df)