import numpy as np
import pandas as pd
from scipy import stats

def detect_outliers_iqr(data, column, threshold=1.5):
    """
    Detect outliers using Interquartile Range method.
    Returns boolean mask for outliers.
    """
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR
    return (data[column] < lower_bound) | (data[column] > upper_bound)

def remove_outliers(data, columns, threshold=1.5):
    """
    Remove outliers from specified columns using IQR method.
    Returns cleaned DataFrame.
    """
    clean_data = data.copy()
    for col in columns:
        outliers = detect_outliers_iqr(clean_data, col, threshold)
        clean_data = clean_data[~outliers]
    return clean_data.reset_index(drop=True)

def normalize_minmax(data, columns):
    """
    Apply Min-Max normalization to specified columns.
    Returns DataFrame with normalized columns.
    """
    normalized_data = data.copy()
    for col in columns:
        min_val = normalized_data[col].min()
        max_val = normalized_data[col].max()
        normalized_data[col] = (normalized_data[col] - min_val) / (max_val - min_val)
    return normalized_data

def normalize_zscore(data, columns):
    """
    Apply Z-score normalization to specified columns.
    Returns DataFrame with normalized columns.
    """
    normalized_data = data.copy()
    for col in columns:
        mean_val = normalized_data[col].mean()
        std_val = normalized_data[col].std()
        normalized_data[col] = (normalized_data[col] - mean_val) / std_val
    return normalized_data

def handle_missing_values(data, strategy='mean', columns=None):
    """
    Handle missing values using specified strategy.
    Supported strategies: 'mean', 'median', 'mode', 'drop'
    """
    if columns is None:
        columns = data.columns
    
    processed_data = data.copy()
    
    if strategy == 'drop':
        return processed_data.dropna(subset=columns)
    
    for col in columns:
        if processed_data[col].isnull().any():
            if strategy == 'mean':
                fill_value = processed_data[col].mean()
            elif strategy == 'median':
                fill_value = processed_data[col].median()
            elif strategy == 'mode':
                fill_value = processed_data[col].mode()[0]
            else:
                raise ValueError(f"Unsupported strategy: {strategy}")
            
            processed_data[col] = processed_data[col].fillna(fill_value)
    
    return processed_data

def validate_dataframe(df, required_columns=None, numeric_columns=None):
    """
    Validate DataFrame structure and data types.
    Returns validation result dictionary.
    """
    validation_result = {
        'is_valid': True,
        'errors': [],
        'warnings': []
    }
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            validation_result['is_valid'] = False
            validation_result['errors'].append(f"Missing required columns: {missing_cols}")
    
    if numeric_columns:
        non_numeric = []
        for col in numeric_columns:
            if col in df.columns and not np.issubdtype(df[col].dtype, np.number):
                non_numeric.append(col)
        
        if non_numeric:
            validation_result['warnings'].append(f"Non-numeric data in columns marked as numeric: {non_numeric}")
    
    if df.isnull().any().any():
        null_counts = df.isnull().sum()
        null_cols = null_counts[null_counts > 0].index.tolist()
        validation_result['warnings'].append(f"Missing values found in columns: {null_cols}")
    
    return validation_resultimport pandas as pd
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
        if fill_missing == 'mean':
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        elif fill_missing == 'median':
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        elif fill_missing == 'mode':
            for col in df.columns:
                df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else None)
        elif fill_missing == 'drop':
            df = df.dropna()
        else:
            raise ValueError("fill_missing must be 'mean', 'median', 'mode', or 'drop'")
    
    print(f"Dataset cleaned. Original shape: {original_shape}, New shape: {df.shape}")
    return df

def validate_data(df, required_columns=None, min_rows=1):
    """
    Validate the dataset structure and content.
    """
    if df.empty:
        raise ValueError("DataFrame is empty.")
    
    if len(df) < min_rows:
        raise ValueError(f"DataFrame has fewer than {min_rows} rows.")
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
    
    print("Data validation passed.")
    return True

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'A': [1, 2, 2, 4, None],
        'B': [5, None, 7, 8, 9],
        'C': ['x', 'y', 'y', 'z', 'x']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    
    cleaned_df = clean_dataset(df, fill_missing='mean')
    print("\nCleaned DataFrame:")
    print(cleaned_df)
    
    validate_data(cleaned_df, required_columns=['A', 'B', 'C'])
import pandas as pd
import re

def clean_dataframe(df, columns_to_clean=None, remove_duplicates=True, case_normalization='lower'):
    """
    Clean a pandas DataFrame by removing duplicates and normalizing string columns.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame.
    columns_to_clean (list, optional): List of column names to clean. If None, clean all object dtype columns.
    remove_duplicates (bool): If True, remove duplicate rows.
    case_normalization (str): One of 'lower', 'upper', or None to specify case normalization.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    df_clean = df.copy()
    
    if columns_to_clean is None:
        columns_to_clean = df_clean.select_dtypes(include=['object']).columns.tolist()
    
    for col in columns_to_clean:
        if col in df_clean.columns and df_clean[col].dtype == 'object':
            df_clean[col] = df_clean[col].astype(str).str.strip()
            
            if case_normalization == 'lower':
                df_clean[col] = df_clean[col].str.lower()
            elif case_normalization == 'upper':
                df_clean[col] = df_clean[col].str.upper()
            
            df_clean[col] = df_clean[col].replace(r'^\s*$', pd.NA, regex=True)
    
    if remove_duplicates:
        initial_rows = len(df_clean)
        df_clean = df_clean.drop_duplicates().reset_index(drop=True)
        removed = initial_rows - len(df_clean)
        print(f"Removed {removed} duplicate rows.")
    
    return df_clean

def normalize_text(text, remove_special=True, replace_whitespace=True):
    """
    Normalize a text string by removing special characters and extra whitespace.
    
    Parameters:
    text (str): Input text.
    remove_special (bool): If True, remove non-alphanumeric characters.
    replace_whitespace (bool): If True, replace multiple whitespaces with single space.
    
    Returns:
    str: Normalized text.
    """
    if not isinstance(text, str):
        return text
    
    normalized = text.strip()
    
    if remove_special:
        normalized = re.sub(r'[^a-zA-Z0-9\s]', '', normalized)
    
    if replace_whitespace:
        normalized = re.sub(r'\s+', ' ', normalized)
    
    return normalized

def validate_email(email):
    """
    Validate an email address format.
    
    Parameters:
    email (str): Email address to validate.
    
    Returns:
    bool: True if email format is valid, False otherwise.
    """
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, str(email)))