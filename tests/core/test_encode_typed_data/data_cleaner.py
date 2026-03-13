
import pandas as pd

def remove_duplicates(dataframe, subset=None, keep='first'):
    """
    Remove duplicate rows from a pandas DataFrame.
    
    Args:
        dataframe: Input pandas DataFrame
        subset: Column label or sequence of labels to consider for duplicates
        keep: Determines which duplicates to keep ('first', 'last', False)
    
    Returns:
        DataFrame with duplicates removed
    """
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    
    cleaned_df = dataframe.drop_duplicates(subset=subset, keep=keep)
    
    removed_count = len(dataframe) - len(cleaned_df)
    if removed_count > 0:
        print(f"Removed {removed_count} duplicate row(s)")
    
    return cleaned_df

def clean_numeric_columns(dataframe, columns=None):
    """
    Clean numeric columns by converting to appropriate types and handling errors.
    
    Args:
        dataframe: Input pandas DataFrame
        columns: List of column names to clean (defaults to all numeric columns)
    
    Returns:
        DataFrame with cleaned numeric columns
    """
    if columns is None:
        columns = dataframe.select_dtypes(include=['object']).columns
    
    for col in columns:
        if col in dataframe.columns:
            try:
                dataframe[col] = pd.to_numeric(dataframe[col], errors='coerce')
            except Exception as e:
                print(f"Could not convert column '{col}': {e}")
    
    return dataframe

def validate_dataframe(dataframe, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Args:
        dataframe: Input pandas DataFrame
        required_columns: List of column names that must be present
    
    Returns:
        Tuple of (is_valid, validation_message)
    """
    if not isinstance(dataframe, pd.DataFrame):
        return False, "Input is not a pandas DataFrame"
    
    if len(dataframe) == 0:
        return False, "DataFrame is empty"
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in dataframe.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
    
    return True, "DataFrame validation passed"