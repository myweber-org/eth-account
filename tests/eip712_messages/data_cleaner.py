import pandas as pd
import numpy as np

def load_and_clean_csv(filepath, drop_na=True, fill_value=None):
    """
    Load a CSV file and perform basic cleaning operations.
    
    Args:
        filepath (str): Path to the CSV file.
        drop_na (bool): If True, drop rows with any NaN values.
        fill_value: If provided and drop_na is False, fill NaN with this value.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {filepath}")
    
    if drop_na:
        df = df.dropna()
    elif fill_value is not None:
        df = df.fillna(fill_value)
    
    return df

def remove_duplicates(df, subset=None):
    """
    Remove duplicate rows from DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        subset (list): Columns to consider for identifying duplicates.
    
    Returns:
        pd.DataFrame: DataFrame with duplicates removed.
    """
    return df.drop_duplicates(subset=subset, keep='first')

def normalize_column(df, column_name):
    """
    Normalize a numeric column to range [0, 1].
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        column_name (str): Name of column to normalize.
    
    Returns:
        pd.DataFrame: DataFrame with normalized column.
    """
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame")
    
    col = df[column_name]
    if not np.issubdtype(col.dtype, np.number):
        raise TypeError(f"Column '{column_name}' must be numeric")
    
    min_val = col.min()
    max_val = col.max()
    
    if max_val == min_val:
        df[column_name] = 0.5
    else:
        df[column_name] = (col - min_val) / (max_val - min_val)
    
    return df

def filter_by_quantile(df, column_name, lower=0.05, upper=0.95):
    """
    Filter DataFrame rows based on column quantile thresholds.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        column_name (str): Column name for filtering.
        lower (float): Lower quantile threshold.
        upper (float): Upper quantile threshold.
    
    Returns:
        pd.DataFrame: Filtered DataFrame.
    """
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame")
    
    col = df[column_name]
    if not np.issubdtype(col.dtype, np.number):
        raise TypeError(f"Column '{column_name}' must be numeric")
    
    lower_bound = col.quantile(lower)
    upper_bound = col.quantile(upper)
    
    return df[(col >= lower_bound) & (col <= upper_bound)]