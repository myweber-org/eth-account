
import pandas as pd
import re

def clean_dataframe(df, text_column):
    """
    Clean a DataFrame by removing duplicate rows and standardizing text in a specified column.
    """
    # Remove duplicate rows
    df_cleaned = df.drop_duplicates().reset_index(drop=True)
    
    # Standardize text: lowercase and remove extra whitespace
    if text_column in df_cleaned.columns:
        df_cleaned[text_column] = df_cleaned[text_column].apply(
            lambda x: re.sub(r'\s+', ' ', str(x).strip().lower())
        )
    
    return df_cleaned

def filter_by_keyword(df, text_column, keyword):
    """
    Filter rows where the specified text column contains a given keyword.
    """
    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in DataFrame")
    
    filtered_df = df[df[text_column].str.contains(keyword, case=False, na=False)]
    return filtered_df.reset_index(drop=True)