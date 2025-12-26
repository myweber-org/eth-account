
import pandas as pd

def remove_duplicates(df, subset=None, keep='first'):
    """
    Remove duplicate rows from a DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        subset (list, optional): Column labels to consider for duplicates
        keep (str, optional): Which duplicates to keep ('first', 'last', False)
    
    Returns:
        pd.DataFrame: DataFrame with duplicates removed
    """
    if df.empty:
        return df
    
    cleaned_df = df.drop_duplicates(subset=subset, keep=keep)
    
    removed_count = len(df) - len(cleaned_df)
    if removed_count > 0:
        print(f"Removed {removed_count} duplicate rows")
    
    return cleaned_df.reset_index(drop=True)

def clean_numeric_columns(df, columns):
    """
    Clean numeric columns by removing non-numeric characters and converting to float.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        columns (list): List of column names to clean
    
    Returns:
        pd.DataFrame: DataFrame with cleaned numeric columns
    """
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace('[^0-9.-]', '', regex=True), errors='coerce')
    
    return df

def standardize_text(df, columns):
    """
    Standardize text columns by trimming whitespace and converting to lowercase.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        columns (list): List of column names to standardize
    
    Returns:
        pd.DataFrame: DataFrame with standardized text columns
    """
    for col in columns:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.lower()
    
    return df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        required_columns (list, optional): List of required column names
    
    Returns:
        tuple: (is_valid, error_message)
    """
    if df.empty:
        return False, "DataFrame is empty"
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
    
    return True, "DataFrame is valid"import pandas as pd

def remove_duplicates(df, subset=None, keep='first'):
    """
    Remove duplicate rows from a DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        subset (list, optional): Column labels to consider for duplicates.
        keep (str, optional): Which duplicates to keep.
    
    Returns:
        pd.DataFrame: DataFrame with duplicates removed.
    """
    if df.empty:
        return df
    
    cleaned_df = df.drop_duplicates(subset=subset, keep=keep)
    return cleaned_df

def validate_dataframe(df):
    """
    Basic validation of DataFrame structure.
    
    Args:
        df (pd.DataFrame): DataFrame to validate.
    
    Returns:
        bool: True if valid, False otherwise.
    """
    if not isinstance(df, pd.DataFrame):
        return False
    if df.shape[0] == 0:
        return False
    return True

def clean_numeric_columns(df, columns):
    """
    Clean numeric columns by converting to appropriate types.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        columns (list): List of column names to clean.
    
    Returns:
        pd.DataFrame: DataFrame with cleaned numeric columns.
    """
    cleaned_df = df.copy()
    for col in columns:
        if col in cleaned_df.columns:
            cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='coerce')
    return cleaned_df
import pandas as pd
import numpy as np

def clean_dataset(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    """
    original_shape = df.shape
    
    if drop_duplicates:
        df = df.drop_duplicates()
        print(f"Removed {original_shape[0] - df.shape[0]} duplicate rows.")
    
    if fill_missing:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if fill_missing == 'mean':
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        elif fill_missing == 'median':
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        elif fill_missing == 'zero':
            df[numeric_cols] = df[numeric_cols].fillna(0)
        
        non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns
        df[non_numeric_cols] = df[non_numeric_cols].fillna('Unknown')
    
    print(f"Dataset cleaned. Original shape: {original_shape}, New shape: {df.shape}")
    return df

def validate_data(df, required_columns=None, min_rows=1):
    """
    Validate the dataset for required columns and minimum rows.
    """
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
    
    if len(df) < min_rows:
        raise ValueError(f"Dataset must have at least {min_rows} rows, but has {len(df)}")
    
    return True

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'id': [1, 2, 2, 3, 4, 5],
        'value': [10.5, 20.3, 20.3, np.nan, 40.1, 50.0],
        'category': ['A', 'B', 'B', 'C', None, 'A']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    
    cleaned_df = clean_dataset(df, drop_duplicates=True, fill_missing='mean')
    print("\nCleaned DataFrame:")
    print(cleaned_df)
    
    try:
        validate_data(cleaned_df, required_columns=['id', 'value'], min_rows=3)
        print("\nData validation passed.")
    except ValueError as e:
        print(f"\nData validation failed: {e}")import numpy as np
import pandas as pd

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range method.
    
    Args:
        df: pandas DataFrame
        column: Column name to process
    
    Returns:
        DataFrame with outliers removed
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
    return filtered_df

def calculate_summary_statistics(df, column):
    """
    Calculate summary statistics for a DataFrame column.
    
    Args:
        df: pandas DataFrame
        column: Column name to analyze
    
    Returns:
        Dictionary containing summary statistics
    """
    stats = {
        'mean': df[column].mean(),
        'median': df[column].median(),
        'std': df[column].std(),
        'min': df[column].min(),
        'max': df[column].max(),
        'count': df[column].count()
    }
    
    return stats

def process_dataframe(df, column):
    """
    Process a DataFrame by removing outliers and calculating statistics.
    
    Args:
        df: pandas DataFrame
        column: Column name to process
    
    Returns:
        Tuple of (cleaned_df, statistics_dict)
    """
    cleaned_df = remove_outliers_iqr(df, column)
    stats = calculate_summary_statistics(cleaned_df, column)
    
    return cleaned_df, stats