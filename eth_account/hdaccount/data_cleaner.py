
import pandas as pd
import re

def clean_text_column(series):
    """
    Standardize text: lowercase, strip whitespace, remove extra spaces.
    """
    if series.dtype == 'object':
        series = series.astype(str)
        series = series.str.lower()
        series = series.str.strip()
        series = series.apply(lambda x: re.sub(r'\s+', ' ', x))
    return series

def remove_duplicates(df, subset=None):
    """
    Remove duplicate rows from DataFrame.
    """
    return df.drop_duplicates(subset=subset, keep='first')

def clean_dataframe(df, text_columns=None):
    """
    Apply cleaning functions to DataFrame.
    """
    df_clean = df.copy()
    
    if text_columns is None:
        text_columns = df_clean.select_dtypes(include=['object']).columns
    
    for col in text_columns:
        df_clean[col] = clean_text_column(df_clean[col])
    
    df_clean = remove_duplicates(df_clean)
    
    return df_clean

if __name__ == "__main__":
    sample_data = {
        'name': ['  John Doe  ', 'Jane Smith', 'JOHN DOE', 'Jane Smith '],
        'age': [25, 30, 25, 30],
        'email': ['JOHN@EXAMPLE.COM', 'jane@example.com', 'john@example.com', 'JANE@example.com']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nCleaned DataFrame:")
    cleaned_df = clean_dataframe(df)
    print(cleaned_df)