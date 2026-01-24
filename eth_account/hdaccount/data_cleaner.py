import re
import string

def remove_punctuation(text):
    """Remove all punctuation from the input string."""
    return text.translate(str.maketrans('', '', string.punctuation))

def normalize_whitespace(text):
    """Replace multiple whitespace characters with a single space."""
    return re.sub(r'\s+', ' ', text).strip()

def clean_text(text, remove_punct=True, normalize_ws=True):
    """Clean text by optionally removing punctuation and normalizing whitespace."""
    if remove_punct:
        text = remove_punctuation(text)
    if normalize_ws:
        text = normalize_whitespace(text)
    return text.lower()

def tokenize_text(text, delimiter=' '):
    """Split text into tokens based on the specified delimiter."""
    return text.split(delimiter)

def process_text_list(text_list, **kwargs):
    """Apply cleaning functions to a list of text strings."""
    return [clean_text(text, **kwargs) for text in text_list]import pandas as pd
import numpy as np

def remove_missing_rows(df, columns=None):
    """
    Remove rows with missing values from DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        columns (list, optional): Specific columns to check for missing values.
                                 If None, checks all columns.
    
    Returns:
        pd.DataFrame: DataFrame with rows containing missing values removed
    """
    if columns is None:
        return df.dropna()
    else:
        return df.dropna(subset=columns)

def fill_missing_with_mean(df, columns=None):
    """
    Fill missing values with column mean.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        columns (list, optional): Specific columns to fill.
                                 If None, fills all numeric columns.
    
    Returns:
        pd.DataFrame: DataFrame with missing values filled
    """
    df_copy = df.copy()
    
    if columns is None:
        numeric_cols = df_copy.select_dtypes(include=[np.number]).columns
        columns = list(numeric_cols)
    
    for col in columns:
        if col in df_copy.columns and pd.api.types.is_numeric_dtype(df_copy[col]):
            mean_val = df_copy[col].mean()
            df_copy[col] = df_copy[col].fillna(mean_val)
    
    return df_copy

def remove_outliers_iqr(df, column, multiplier=1.5):
    """
    Remove outliers using IQR method.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        column (str): Column name to check for outliers
        multiplier (float): IQR multiplier for outlier detection
    
    Returns:
        pd.DataFrame: DataFrame with outliers removed
    """
    if column not in df.columns:
        return df
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def standardize_columns(df, columns=None):
    """
    Standardize numeric columns to have zero mean and unit variance.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        columns (list, optional): Specific columns to standardize.
                                 If None, standardizes all numeric columns.
    
    Returns:
        pd.DataFrame: DataFrame with standardized columns
    """
    df_copy = df.copy()
    
    if columns is None:
        numeric_cols = df_copy.select_dtypes(include=[np.number]).columns
        columns = list(numeric_cols)
    
    for col in columns:
        if col in df_copy.columns and pd.api.types.is_numeric_dtype(df_copy[col]):
            mean_val = df_copy[col].mean()
            std_val = df_copy[col].std()
            
            if std_val > 0:
                df_copy[col] = (df_copy[col] - mean_val) / std_val
    
    return df_copy

def clean_dataset(df, missing_strategy='remove', outlier_columns=None):
    """
    Comprehensive data cleaning function.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        missing_strategy (str): Strategy for handling missing values.
                               Options: 'remove', 'mean'
        outlier_columns (list, optional): Columns to check for outliers.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    # Handle missing values
    if missing_strategy == 'remove':
        cleaned_df = remove_missing_rows(cleaned_df)
    elif missing_strategy == 'mean':
        cleaned_df = fill_missing_with_mean(cleaned_df)
    
    # Remove outliers if specified
    if outlier_columns:
        for col in outlier_columns:
            if col in cleaned_df.columns:
                cleaned_df = remove_outliers_iqr(cleaned_df, col)
    
    # Standardize numeric columns
    cleaned_df = standardize_columns(cleaned_df)
    
    return cleaned_dfimport pandas as pd
import numpy as np

def remove_missing_rows(df, columns=None):
    """
    Remove rows with missing values from the DataFrame.
    
    Args:
        df: pandas DataFrame
        columns: list of columns to check for missing values. 
                 If None, checks all columns.
    
    Returns:
        Cleaned DataFrame
    """
    if columns is None:
        columns = df.columns
    
    return df.dropna(subset=columns)

def fill_missing_with_mean(df, columns=None):
    """
    Fill missing values with column mean.
    
    Args:
        df: pandas DataFrame
        columns: list of columns to fill. If None, fills all numeric columns.
    
    Returns:
        DataFrame with filled values
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    df_filled = df.copy()
    for col in columns:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            df_filled[col] = df[col].fillna(df[col].mean())
    
    return df_filled

def detect_outliers_iqr(df, column, threshold=1.5):
    """
    Detect outliers using IQR method.
    
    Args:
        df: pandas DataFrame
        column: column name to check for outliers
        threshold: IQR multiplier (default: 1.5)
    
    Returns:
        Boolean Series indicating outliers
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    if not pd.api.types.is_numeric_dtype(df[column]):
        raise ValueError(f"Column '{column}' must be numeric")
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR
    
    return (df[column] < lower_bound) | (df[column] > upper_bound)

def remove_outliers(df, column, threshold=1.5):
    """
    Remove outliers from a DataFrame column.
    
    Args:
        df: pandas DataFrame
        column: column name to remove outliers from
        threshold: IQR multiplier (default: 1.5)
    
    Returns:
        DataFrame without outliers
    """
    outliers = detect_outliers_iqr(df, column, threshold)
    return df[~outliers]

def standardize_columns(df, columns=None):
    """
    Standardize numeric columns to have mean=0 and std=1.
    
    Args:
        df: pandas DataFrame
        columns: list of columns to standardize. If None, standardizes all numeric columns.
    
    Returns:
        DataFrame with standardized columns
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    df_standardized = df.copy()
    for col in columns:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            mean = df[col].mean()
            std = df[col].std()
            if std > 0:
                df_standardized[col] = (df[col] - mean) / std
    
    return df_standardized

def clean_dataset(df, missing_strategy='remove', outlier_columns=None, standardize=False):
    """
    Comprehensive data cleaning pipeline.
    
    Args:
        df: pandas DataFrame
        missing_strategy: 'remove' or 'mean' for handling missing values
        outlier_columns: list of columns to remove outliers from
        standardize: whether to standardize numeric columns
    
    Returns:
        Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    # Handle missing values
    if missing_strategy == 'remove':
        cleaned_df = remove_missing_rows(cleaned_df)
    elif missing_strategy == 'mean':
        cleaned_df = fill_missing_with_mean(cleaned_df)
    
    # Remove outliers
    if outlier_columns:
        for col in outlier_columns:
            if col in cleaned_df.columns:
                cleaned_df = remove_outliers(cleaned_df, col)
    
    # Standardize
    if standardize:
        cleaned_df = standardize_columns(cleaned_df)
    
    return cleaned_df
import pandas as pd
import numpy as np

def clean_dataset(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    drop_duplicates (bool): Whether to remove duplicate rows.
    fill_missing (str): Method to fill missing values ('mean', 'median', 'mode', or 'drop').
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        removed = initial_rows - len(cleaned_df)
        print(f"Removed {removed} duplicate rows.")
    
    if fill_missing == 'drop':
        cleaned_df = cleaned_df.dropna()
        print("Dropped rows with missing values.")
    elif fill_missing in ['mean', 'median', 'mode']:
        numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if cleaned_df[col].isnull().any():
                if fill_missing == 'mean':
                    fill_value = cleaned_df[col].mean()
                elif fill_missing == 'median':
                    fill_value = cleaned_df[col].median()
                else:  # mode
                    fill_value = cleaned_df[col].mode()[0]
                cleaned_df[col].fillna(fill_value, inplace=True)
                print(f"Filled missing values in column '{col}' with {fill_missing}: {fill_value}")
    
    categorical_cols = cleaned_df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if cleaned_df[col].isnull().any():
            cleaned_df[col].fillna('Unknown', inplace=True)
            print(f"Filled missing values in categorical column '{col}' with 'Unknown'")
    
    return cleaned_df

def validate_data(df, required_columns=None):
    """
    Validate the DataFrame for required columns and data types.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of required column names.
    
    Returns:
    bool: True if validation passes, False otherwise.
    """
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            print(f"Missing required columns: {missing_cols}")
            return False
    
    if df.empty:
        print("DataFrame is empty.")
        return False
    
    print("Data validation passed.")
    return True

if __name__ == "__main__":
    sample_data = {
        'id': [1, 2, 3, 4, 4, 5],
        'value': [10.5, np.nan, 15.2, 10.5, 12.0, np.nan],
        'category': ['A', 'B', 'A', np.nan, 'C', 'B'],
        'score': [85, 92, 78, 85, 90, 92]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    cleaned = clean_dataset(df, drop_duplicates=True, fill_missing='mean')
    print("\nCleaned DataFrame:")
    print(cleaned)
    
    is_valid = validate_data(cleaned, required_columns=['id', 'value', 'category'])
    print(f"\nData is valid: {is_valid}")