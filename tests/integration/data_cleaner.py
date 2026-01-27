
import pandas as pd

def remove_duplicates(dataframe, subset=None, keep='first'):
    """
    Remove duplicate rows from a pandas DataFrame.
    
    Args:
        dataframe: Input pandas DataFrame.
        subset: Column label or sequence of labels to consider for identifying duplicates.
                If None, all columns are used.
        keep: Determines which duplicates to mark.
              'first' : Mark duplicates as True except for the first occurrence.
              'last' : Mark duplicates as True except for the last occurrence.
              False : Mark all duplicates as True.
    
    Returns:
        DataFrame with duplicates removed.
    """
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    
    cleaned_df = dataframe.drop_duplicates(subset=subset, keep=keep)
    return cleaned_df

def clean_numeric_columns(dataframe, columns):
    """
    Clean numeric columns by converting to appropriate type and handling errors.
    
    Args:
        dataframe: Input pandas DataFrame.
        columns: List of column names to clean.
    
    Returns:
        DataFrame with cleaned numeric columns.
    """
    for col in columns:
        if col in dataframe.columns:
            dataframe[col] = pd.to_numeric(dataframe[col], errors='coerce')
    return dataframe

def validate_dataframe(dataframe, required_columns):
    """
    Validate that DataFrame contains all required columns.
    
    Args:
        dataframe: Input pandas DataFrame.
        required_columns: List of required column names.
    
    Returns:
        Boolean indicating if all required columns are present.
    """
    missing_columns = [col for col in required_columns if col not in dataframe.columns]
    
    if missing_columns:
        print(f"Missing columns: {missing_columns}")
        return False
    return True