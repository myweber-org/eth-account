
import pandas as pd

def clean_dataset(df, drop_na=True, rename_columns=True):
    """
    Clean a pandas DataFrame by handling missing values and standardizing column names.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    drop_na (bool): If True, drop rows with any null values. Default True.
    rename_columns (bool): If True, rename columns to lowercase with underscores. Default True.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_na:
        cleaned_df = cleaned_df.dropna()
    
    if rename_columns:
        cleaned_df.columns = (
            cleaned_df.columns
            .str.lower()
            .str.replace(r'[^a-z0-9]+', '_', regex=True)
            .str.strip('_')
        )
    
    return cleaned_df

def validate_dataset(df, required_columns=None):
    """
    Validate a DataFrame for required columns and data types.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of required column names. Default None.
    
    Returns:
    bool: True if validation passes, False otherwise.
    """
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Missing required columns: {missing_columns}")
            return False
    
    if df.empty:
        print("DataFrame is empty.")
        return False
    
    return True

if __name__ == "__main__":
    sample_data = {
        'First Name': ['Alice', 'Bob', None],
        'Last Name': ['Smith', None, 'Johnson'],
        'Age': [25, 30, 35]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nCleaned DataFrame:")
    cleaned = clean_dataset(df)
    print(cleaned)
    print(f"\nValidation passed: {validate_dataset(cleaned)}")
import pandas as pd
import numpy as np

def clean_dataframe(df, missing_strategy='mean', outlier_method='iqr', columns=None):
    """
    Clean a pandas DataFrame by handling missing values and outliers.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    missing_strategy (str): Strategy for missing values. Options: 'mean', 'median', 'mode', 'drop'.
    outlier_method (str): Method for outlier detection. Options: 'iqr', 'zscore'.
    columns (list): List of column names to clean. If None, clean all numeric columns.

    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    df_clean = df.copy()

    if columns is None:
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
        columns = numeric_cols

    for col in columns:
        if col not in df_clean.columns:
            continue

        if missing_strategy == 'mean':
            fill_value = df_clean[col].mean()
        elif missing_strategy == 'median':
            fill_value = df_clean[col].median()
        elif missing_strategy == 'mode':
            fill_value = df_clean[col].mode()[0] if not df_clean[col].mode().empty else np.nan
        elif missing_strategy == 'drop':
            df_clean = df_clean.dropna(subset=[col])
            continue
        else:
            raise ValueError(f"Unsupported missing strategy: {missing_strategy}")

        df_clean[col].fillna(fill_value, inplace=True)

        if outlier_method == 'iqr':
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df_clean[col] = np.where((df_clean[col] < lower_bound) | (df_clean[col] > upper_bound),
                                     fill_value, df_clean[col])
        elif outlier_method == 'zscore':
            mean = df_clean[col].mean()
            std = df_clean[col].std()
            z_scores = np.abs((df_clean[col] - mean) / std)
            df_clean[col] = np.where(z_scores > 3, fill_value, df_clean[col])
        else:
            raise ValueError(f"Unsupported outlier method: {outlier_method}")

    return df_cleanimport pandas as pd
import numpy as np

def clean_missing_data(df, strategy='mean', columns=None):
    """
    Clean missing data in a DataFrame using specified strategy.
    
    Args:
        df: pandas DataFrame containing data with potential missing values
        strategy: Method for handling missing values ('mean', 'median', 'mode', 'drop')
        columns: List of column names to apply cleaning to (None for all columns)
    
    Returns:
        Cleaned pandas DataFrame
    """
    if df.empty:
        return df
    
    if columns is None:
        columns = df.columns
    
    df_clean = df.copy()
    
    for col in columns:
        if col not in df_clean.columns:
            continue
            
        if df_clean[col].isnull().any():
            if strategy == 'mean':
                if pd.api.types.is_numeric_dtype(df_clean[col]):
                    df_clean[col].fillna(df_clean[col].mean(), inplace=True)
            elif strategy == 'median':
                if pd.api.types.is_numeric_dtype(df_clean[col]):
                    df_clean[col].fillna(df_clean[col].median(), inplace=True)
            elif strategy == 'mode':
                df_clean[col].fillna(df_clean[col].mode()[0] if not df_clean[col].mode().empty else np.nan, inplace=True)
            elif strategy == 'drop':
                df_clean = df_clean.dropna(subset=[col])
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
    
    return df_clean

def detect_outliers_iqr(df, column, threshold=1.5):
    """
    Detect outliers using Interquartile Range method.
    
    Args:
        df: pandas DataFrame
        column: Column name to check for outliers
        threshold: IQR multiplier for outlier detection
    
    Returns:
        Boolean mask indicating outliers
    """
    if not pd.api.types.is_numeric_dtype(df[column]):
        return pd.Series([False] * len(df))
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR
    
    return (df[column] < lower_bound) | (df[column] > upper_bound)

def normalize_column(df, column, method='minmax'):
    """
    Normalize a column in the DataFrame.
    
    Args:
        df: pandas DataFrame
        column: Column name to normalize
        method: Normalization method ('minmax' or 'zscore')
    
    Returns:
        DataFrame with normalized column
    """
    if not pd.api.types.is_numeric_dtype(df[column]):
        return df
    
    df_normalized = df.copy()
    
    if method == 'minmax':
        min_val = df[column].min()
        max_val = df[column].max()
        if max_val != min_val:
            df_normalized[column] = (df[column] - min_val) / (max_val - min_val)
    
    elif method == 'zscore':
        mean_val = df[column].mean()
        std_val = df[column].std()
        if std_val != 0:
            df_normalized[column] = (df[column] - mean_val) / std_val
    
    return df_normalized

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, np.nan, 4, 5],
        'B': [10, 20, 30, np.nan, 50],
        'C': [100, 200, 300, 400, 500]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    
    cleaned_df = clean_missing_data(df, strategy='mean')
    print("\nCleaned DataFrame (mean imputation):")
    print(cleaned_df)
    
    outliers = detect_outliers_iqr(cleaned_df, 'A')
    print(f"\nOutliers in column A: {outliers.sum()}")
    
    normalized_df = normalize_column(cleaned_df, 'C', method='minmax')
    print("\nNormalized column C:")
    print(normalized_df[['C']])import pandas as pd
import numpy as np

def remove_duplicates(df, subset=None):
    """
    Remove duplicate rows from DataFrame.
    
    Args:
        df: pandas DataFrame
        subset: column label or sequence of labels to consider for duplicates
    
    Returns:
        DataFrame with duplicates removed
    """
    return df.drop_duplicates(subset=subset, keep='first')

def convert_column_types(df, column_type_map):
    """
    Convert specified columns to given data types.
    
    Args:
        df: pandas DataFrame
        column_type_map: dict mapping column names to target types
    
    Returns:
        DataFrame with converted column types
    """
    df_converted = df.copy()
    for column, dtype in column_type_map.items():
        if column in df_converted.columns:
            df_converted[column] = df_converted[column].astype(dtype)
    return df_converted

def handle_missing_values(df, strategy='drop', fill_value=None):
    """
    Handle missing values in DataFrame.
    
    Args:
        df: pandas DataFrame
        strategy: 'drop' to remove rows, 'fill' to fill values
        fill_value: value to use when strategy is 'fill'
    
    Returns:
        DataFrame with handled missing values
    """
    if strategy == 'drop':
        return df.dropna()
    elif strategy == 'fill':
        return df.fillna(fill_value)
    else:
        raise ValueError("Strategy must be 'drop' or 'fill'")

def clean_dataframe(df, deduplicate=True, type_conversions=None, missing_strategy='drop'):
    """
    Comprehensive data cleaning function.
    
    Args:
        df: pandas DataFrame
        deduplicate: whether to remove duplicates
        type_conversions: dict for column type conversions
        missing_strategy: strategy for handling missing values
    
    Returns:
        Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    if deduplicate:
        cleaned_df = remove_duplicates(cleaned_df)
    
    if type_conversions:
        cleaned_df = convert_column_types(cleaned_df, type_conversions)
    
    cleaned_df = handle_missing_values(cleaned_df, strategy=missing_strategy)
    
    return cleaned_df

def validate_dataframe(df, required_columns=None, min_rows=1):
    """
    Validate DataFrame structure and content.
    
    Args:
        df: pandas DataFrame
        required_columns: list of columns that must be present
        min_rows: minimum number of rows required
    
    Returns:
        tuple of (is_valid, error_message)
    """
    if df.empty:
        return False, "DataFrame is empty"
    
    if len(df) < min_rows:
        return False, f"DataFrame has fewer than {min_rows} rows"
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
    
    return True, "DataFrame is valid"