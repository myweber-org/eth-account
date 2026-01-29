
import pandas as pd
import numpy as np

def remove_duplicates(df, subset=None, keep='first'):
    """
    Remove duplicate rows from DataFrame.
    
    Args:
        df: pandas DataFrame
        subset: column label or sequence of labels to consider for duplicates
        keep: determines which duplicates to mark ('first', 'last', False)
    
    Returns:
        DataFrame with duplicates removed
    """
    return df.drop_duplicates(subset=subset, keep=keep)

def handle_missing_values(df, strategy='drop', fill_value=None):
    """
    Handle missing values in DataFrame.
    
    Args:
        df: pandas DataFrame
        strategy: 'drop' to remove rows/cols, 'fill' to replace values
        fill_value: value to use when strategy is 'fill'
    
    Returns:
        DataFrame with handled missing values
    """
    if strategy == 'drop':
        return df.dropna()
    elif strategy == 'fill':
        if fill_value is not None:
            return df.fillna(fill_value)
        else:
            return df.fillna(df.mean())
    else:
        raise ValueError("Strategy must be 'drop' or 'fill'")

def normalize_column(df, column):
    """
    Normalize a column to range [0, 1].
    
    Args:
        df: pandas DataFrame
        column: column name to normalize
    
    Returns:
        DataFrame with normalized column
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    col_min = df[column].min()
    col_max = df[column].max()
    
    if col_max == col_min:
        df[column] = 0.5
    else:
        df[column] = (df[column] - col_min) / (col_max - col_min)
    
    return df

def clean_dataframe(df, deduplicate=True, handle_nulls=True, normalize_cols=None):
    """
    Comprehensive data cleaning pipeline.
    
    Args:
        df: pandas DataFrame
        deduplicate: whether to remove duplicates
        handle_nulls: whether to handle missing values
        normalize_cols: list of columns to normalize
    
    Returns:
        Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    if deduplicate:
        cleaned_df = remove_duplicates(cleaned_df)
    
    if handle_nulls:
        cleaned_df = handle_missing_values(cleaned_df, strategy='fill')
    
    if normalize_cols:
        for col in normalize_cols:
            if col in cleaned_df.columns:
                cleaned_df = normalize_column(cleaned_df, col)
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Args:
        df: pandas DataFrame
        required_columns: list of required column names
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not isinstance(df, pd.DataFrame):
        return False, "Input is not a pandas DataFrame"
    
    if df.empty:
        return False, "DataFrame is empty"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    return True, "DataFrame is valid"

# Example usage
if __name__ == "__main__":
    # Create sample data
    sample_data = {
        'id': [1, 2, 2, 3, 4, 5],
        'value': [10, 20, 20, np.nan, 40, 50],
        'score': [100, 200, 200, 300, 400, 500]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print()
    
    # Clean the data
    cleaned = clean_dataframe(
        df, 
        deduplicate=True, 
        handle_nulls=True, 
        normalize_cols=['value', 'score']
    )
    
    print("Cleaned DataFrame:")
    print(cleaned)
    print()
    
    # Validate
    is_valid, message = validate_dataframe(cleaned, required_columns=['id', 'value'])
    print(f"Validation: {is_valid} - {message}")