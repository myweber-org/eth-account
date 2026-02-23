
def remove_duplicates(sequence):
    seen = set()
    result = []
    for item in sequence:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return resultimport numpy as np
import pandas as pd

def remove_outliers_iqr(data, column, multiplier=1.5):
    """
    Remove outliers from a pandas Series using the IQR method.
    
    Args:
        data: pandas DataFrame
        column: Column name to process
        multiplier: IQR multiplier for outlier detection
    
    Returns:
        DataFrame with outliers removed
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    series = data[column]
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    
    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr
    
    return data[(series >= lower_bound) & (series <= upper_bound)]

def normalize_minmax(data, column):
    """
    Normalize data to [0, 1] range using min-max scaling.
    
    Args:
        data: pandas DataFrame
        column: Column name to normalize
    
    Returns:
        DataFrame with normalized column
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    series = data[column]
    min_val = series.min()
    max_val = series.max()
    
    if max_val == min_val:
        normalized = pd.Series([0.5] * len(series), index=series.index)
    else:
        normalized = (series - min_val) / (max_val - min_val)
    
    result = data.copy()
    result[f"{column}_normalized"] = normalized
    return result

def clean_dataset(df, numeric_columns=None, outlier_multiplier=1.5):
    """
    Clean dataset by removing outliers and normalizing numeric columns.
    
    Args:
        df: Input DataFrame
        numeric_columns: List of numeric column names to process
        outlier_multiplier: IQR multiplier for outlier detection
    
    Returns:
        Cleaned DataFrame
    """
    if df.empty:
        return df
    
    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_df = df.copy()
    
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, col, outlier_multiplier)
            cleaned_df = normalize_minmax(cleaned_df, col)
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Args:
        df: DataFrame to validate
        required_columns: List of required column names
    
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
import pandas as pd

def filter_and_clean_dataframe(df, column, condition_func, drop_na=True):
    """
    Filters a DataFrame based on a condition applied to a specific column,
    optionally drops rows with NaN values in that column, and returns a cleaned copy.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    column (str): The column name to apply the filter on.
    condition_func (function): A function that takes a value and returns a boolean.
    drop_na (bool): If True, drops rows where the specified column is NaN before filtering.

    Returns:
    pd.DataFrame: A new filtered and cleaned DataFrame.
    """
    df_clean = df.copy()

    if drop_na:
        df_clean = df_clean.dropna(subset=[column])

    mask = df_clean[column].apply(condition_func)
    filtered_df = df_clean[mask].reset_index(drop=True)

    return filtered_df
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range method.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        column (str): Column name to process
    
    Returns:
        pd.DataFrame: DataFrame with outliers removed
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
    return filtered_df

def clean_numeric_data(df, columns=None):
    """
    Clean numeric data by removing outliers from specified columns.
    If no columns specified, clean all numeric columns.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        columns (list, optional): List of column names to clean
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_df = df.copy()
    for col in columns:
        if col in cleaned_df.columns and pd.api.types.is_numeric_dtype(cleaned_df[col]):
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
    
    return cleaned_df

def get_cleaning_report(original_df, cleaned_df):
    """
    Generate a report comparing original and cleaned DataFrames.
    
    Args:
        original_df (pd.DataFrame): Original DataFrame
        cleaned_df (pd.DataFrame): Cleaned DataFrame
    
    Returns:
        dict: Dictionary containing cleaning statistics
    """
    report = {
        'original_rows': len(original_df),
        'cleaned_rows': len(cleaned_df),
        'rows_removed': len(original_df) - len(cleaned_df),
        'removal_percentage': ((len(original_df) - len(cleaned_df)) / len(original_df)) * 100
    }
    
    return report
import numpy as np

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a specified column using the IQR method.
    
    Args:
        data (pd.DataFrame): The input DataFrame.
        column (str): The column name to process.
    
    Returns:
        pd.DataFrame: DataFrame with outliers removed.
    """
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data

def calculate_summary_statistics(data, column):
    """
    Calculate summary statistics for a specified column.
    
    Args:
        data (pd.DataFrame): The input DataFrame.
        column (str): The column name to process.
    
    Returns:
        dict: Dictionary containing mean, median, and standard deviation.
    """
    stats = {
        'mean': data[column].mean(),
        'median': data[column].median(),
        'std': data[column].std()
    }
    return stats