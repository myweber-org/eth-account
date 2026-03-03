import pandas as pd

def remove_duplicates(df, subset=None, keep='first'):
    """
    Remove duplicate rows from a DataFrame.
    
    Args:
        df: pandas DataFrame
        subset: column label or sequence of labels to consider for identifying duplicates
        keep: determines which duplicates to keep ('first', 'last', False)
    
    Returns:
        DataFrame with duplicates removed
    """
    if subset is None:
        return df.drop_duplicates(keep=keep)
    else:
        return df.drop_duplicates(subset=subset, keep=keep)

def clean_numeric_columns(df, columns):
    """
    Clean numeric columns by converting to appropriate type and handling errors.
    
    Args:
        df: pandas DataFrame
        columns: list of column names to clean
    
    Returns:
        DataFrame with cleaned numeric columns
    """
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def validate_dataframe(df, required_columns):
    """
    Validate that DataFrame contains all required columns.
    
    Args:
        df: pandas DataFrame
        required_columns: list of required column names
    
    Returns:
        Boolean indicating if all required columns are present
    """
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"Missing columns: {missing_columns}")
        return False
    return True

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'id': [1, 2, 2, 3, 4, 4],
        'name': ['Alice', 'Bob', 'Bob', 'Charlie', 'David', 'David'],
        'score': ['85', '90', '90', '78', '92', '92']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    
    # Remove duplicates
    cleaned_df = remove_duplicates(df, subset=['id', 'name'])
    print("\nDataFrame after removing duplicates:")
    print(cleaned_df)
    
    # Clean numeric columns
    cleaned_df = clean_numeric_columns(cleaned_df, ['score'])
    print("\nDataFrame after cleaning numeric columns:")
    print(cleaned_df)
    
    # Validate columns
    is_valid = validate_dataframe(cleaned_df, ['id', 'name', 'score'])
    print(f"\nDataFrame validation: {is_valid}")
import pandas as pd

def remove_duplicates(df, subset=None, keep='first'):
    """
    Remove duplicate rows from a DataFrame.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    subset (list, optional): Column labels to consider for duplicates
    keep (str, optional): Which duplicates to keep - 'first', 'last', or False
    
    Returns:
    pd.DataFrame: DataFrame with duplicates removed
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    
    return df.drop_duplicates(subset=subset, keep=keep)

def clean_numeric_columns(df, columns):
    """
    Clean numeric columns by converting to numeric and filling NaN with mean.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    columns (list): List of column names to clean
    
    Returns:
    pd.DataFrame: DataFrame with cleaned numeric columns
    """
    cleaned_df = df.copy()
    
    for col in columns:
        if col in cleaned_df.columns:
            cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='coerce')
            col_mean = cleaned_df[col].mean()
            cleaned_df[col] = cleaned_df[col].fillna(col_mean)
    
    return cleaned_df

def validate_dataframe(df, required_columns):
    """
    Validate that DataFrame contains all required columns.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    required_columns (list): List of required column names
    
    Returns:
    bool: True if all required columns are present
    """
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    return True