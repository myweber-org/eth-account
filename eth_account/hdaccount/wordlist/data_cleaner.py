
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]

def normalize_minmax(data, column):
    min_val = data[column].min()
    max_val = data[column].max()
    data[column + '_normalized'] = (data[column] - min_val) / (max_val - min_val)
    return data

def standardize_zscore(data, column):
    mean_val = data[column].mean()
    std_val = data[column].std()
    data[column + '_standardized'] = (data[column] - mean_val) / std_val
    return data

def clean_dataset(df, numeric_columns):
    cleaned_df = df.copy()
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
            cleaned_df = normalize_minmax(cleaned_df, col)
            cleaned_df = standardize_zscore(cleaned_df, col)
    return cleaned_df
import pandas as pd
import re

def clean_text_column(df, column_name):
    """
    Standardize text by converting to lowercase and removing extra whitespace.
    """
    df[column_name] = df[column_name].astype(str).str.lower()
    df[column_name] = df[column_name].apply(lambda x: re.sub(r'\s+', ' ', x).strip())
    return df

def remove_duplicate_rows(df, subset=None):
    """
    Remove duplicate rows from the DataFrame.
    """
    return df.drop_duplicates(subset=subset, keep='first')

def clean_numeric_column(df, column_name, fill_value=0):
    """
    Clean numeric column by filling NaN values and converting to appropriate type.
    """
    df[column_name] = pd.to_numeric(df[column_name], errors='coerce')
    df[column_name] = df[column_name].fillna(fill_value)
    return df

def process_dataframe(df, text_columns=None, numeric_columns=None, duplicate_subset=None):
    """
    Main function to clean the entire DataFrame.
    """
    if text_columns:
        for col in text_columns:
            df = clean_text_column(df, col)
    
    if numeric_columns:
        for col in numeric_columns:
            df = clean_numeric_column(df, col)
    
    if duplicate_subset:
        df = remove_duplicate_rows(df, subset=duplicate_subset)
    
    return df

if __name__ == "__main__":
    sample_data = {
        'name': ['Alice', 'alice', 'Bob  ', 'Charlie', 'CHARLIE'],
        'age': [25, 25, 30, None, 35],
        'city': ['New York', 'new york', 'Los Angeles', 'Chicago', 'CHICAGO']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    
    cleaned_df = process_dataframe(
        df,
        text_columns=['name', 'city'],
        numeric_columns=['age'],
        duplicate_subset=['name', 'city']
    )
    
    print("\nCleaned DataFrame:")
    print(cleaned_df)