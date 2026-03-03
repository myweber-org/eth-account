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
import pandas as pd
import numpy as np

def clean_dataset(df, missing_strategy='mean', remove_duplicates=True):
    """
    Clean a pandas DataFrame by handling missing values and removing duplicates.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean
    missing_strategy (str): Strategy for handling missing values ('mean', 'median', 'mode', 'drop')
    remove_duplicates (bool): Whether to remove duplicate rows
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    
    df_clean = df.copy()
    
    # Handle missing values
    if missing_strategy == 'mean':
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].mean())
    elif missing_strategy == 'median':
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].median())
    elif missing_strategy == 'mode':
        for col in df_clean.columns:
            mode_val = df_clean[col].mode()
            if not mode_val.empty:
                df_clean[col] = df_clean[col].fillna(mode_val.iloc[0])
    elif missing_strategy == 'drop':
        df_clean = df_clean.dropna()
    
    # Remove duplicates
    if remove_duplicates:
        df_clean = df_clean.drop_duplicates()
    
    return df_clean

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate
    required_columns (list): List of required column names
    
    Returns:
    dict: Validation results with status and messages
    """
    validation_result = {
        'is_valid': True,
        'messages': [],
        'missing_columns': []
    }
    
    if not isinstance(df, pd.DataFrame):
        validation_result['is_valid'] = False
        validation_result['messages'].append('Input is not a pandas DataFrame')
        return validation_result
    
    if df.empty:
        validation_result['is_valid'] = False
        validation_result['messages'].append('DataFrame is empty')
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            validation_result['is_valid'] = False
            validation_result['missing_columns'] = missing_cols
            validation_result['messages'].append(f'Missing required columns: {missing_cols}')
    
    return validation_result

def normalize_columns(df, columns=None):
    """
    Normalize specified columns to have values between 0 and 1.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    columns (list): List of column names to normalize
    
    Returns:
    pd.DataFrame: DataFrame with normalized columns
    """
    df_normalized = df.copy()
    
    if columns is None:
        numeric_cols = df_normalized.select_dtypes(include=[np.number]).columns
        columns = list(numeric_cols)
    
    for col in columns:
        if col in df_normalized.columns and pd.api.types.is_numeric_dtype(df_normalized[col]):
            col_min = df_normalized[col].min()
            col_max = df_normalized[col].max()
            
            if col_max > col_min:  # Avoid division by zero
                df_normalized[col] = (df_normalized[col] - col_min) / (col_max - col_min)
    
    return df_normalized

# Example usage
if __name__ == "__main__":
    # Create sample data
    sample_data = {
        'A': [1, 2, np.nan, 4, 5, 5],
        'B': [10, 20, 30, np.nan, 50, 10],
        'C': ['x', 'y', 'z', 'x', 'y', 'x']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n")
    
    # Clean the data
    cleaned_df = clean_dataset(df, missing_strategy='mean', remove_duplicates=True)
    print("Cleaned DataFrame:")
    print(cleaned_df)
    print("\n")
    
    # Validate the cleaned data
    validation = validate_dataframe(cleaned_df, required_columns=['A', 'B', 'C'])
    print("Validation Result:")
    print(validation)
    print("\n")
    
    # Normalize numeric columns
    normalized_df = normalize_columns(cleaned_df)
    print("Normalized DataFrame:")
    print(normalized_df)