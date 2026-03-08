
import pandas as pd

def clean_column_data(df, column_name):
    """
    Clean a specified column in a DataFrame by stripping whitespace and converting to lowercase.
    
    Parameters:
    df (pd.DataFrame): The DataFrame containing the column to clean.
    column_name (str): The name of the column to clean.
    
    Returns:
    pd.DataFrame: A DataFrame with the cleaned column.
    
    Raises:
    ValueError: If the specified column does not exist in the DataFrame.
    """
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame.")
    
    df[column_name] = df[column_name].astype(str).str.strip().str.lower()
    return df