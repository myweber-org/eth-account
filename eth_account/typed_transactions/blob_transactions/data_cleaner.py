
def remove_duplicates(sequence):
    seen = set()
    result = []
    for item in sequence:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result
import pandas as pd

def remove_duplicates(df, subset=None, keep='first'):
    """
    Remove duplicate rows from a DataFrame.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    subset (list, optional): Column labels to consider for identifying duplicates
    keep (str, optional): Which duplicates to keep - 'first', 'last', or False
    
    Returns:
    pd.DataFrame: DataFrame with duplicates removed
    """
    if subset is None:
        subset = df.columns.tolist()
    
    cleaned_df = df.drop_duplicates(subset=subset, keep=keep)
    
    removed_count = len(df) - len(cleaned_df)
    print(f"Removed {removed_count} duplicate rows")
    
    return cleaned_df.reset_index(drop=True)

def clean_numeric_columns(df, columns):
    """
    Clean numeric columns by removing non-numeric characters and converting to float.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    columns (list): List of column names to clean
    
    Returns:
    pd.DataFrame: DataFrame with cleaned numeric columns
    """
    cleaned_df = df.copy()
    
    for col in columns:
        if col in cleaned_df.columns:
            cleaned_df[col] = pd.to_numeric(
                cleaned_df[col].astype(str).str.replace(r'[^\d.-]', '', regex=True),
                errors='coerce'
            )
    
    return cleaned_df

def standardize_text(df, columns):
    """
    Standardize text columns by converting to lowercase and stripping whitespace.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    columns (list): List of column names to standardize
    
    Returns:
    pd.DataFrame: DataFrame with standardized text columns
    """
    cleaned_df = df.copy()
    
    for col in columns:
        if col in cleaned_df.columns:
            cleaned_df[col] = cleaned_df[col].astype(str).str.lower().str.strip()
    
    return cleaned_df

def clean_dataframe(df, duplicate_config=None, numeric_columns=None, text_columns=None):
    """
    Comprehensive data cleaning function.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    duplicate_config (dict, optional): Configuration for duplicate removal
    numeric_columns (list, optional): Columns to clean as numeric
    text_columns (list, optional): Columns to standardize as text
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    if duplicate_config:
        cleaned_df = remove_duplicates(cleaned_df, **duplicate_config)
    
    if numeric_columns:
        cleaned_df = clean_numeric_columns(cleaned_df, numeric_columns)
    
    if text_columns:
        cleaned_df = standardize_text(cleaned_df, text_columns)
    
    return cleaned_df