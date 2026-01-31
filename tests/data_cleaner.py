
import pandas as pd

def clean_dataset(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    drop_duplicates (bool): Whether to drop duplicate rows. Default is True.
    fill_missing (str): Method to fill missing values. Options: 'mean', 'median', 'mode', or 'drop'. Default is 'mean'.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    if fill_missing == 'drop':
        cleaned_df = cleaned_df.dropna()
    elif fill_missing in ['mean', 'median']:
        numeric_cols = cleaned_df.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            if fill_missing == 'mean':
                cleaned_df[col].fillna(cleaned_df[col].mean(), inplace=True)
            elif fill_missing == 'median':
                cleaned_df[col].fillna(cleaned_df[col].median(), inplace=True)
    elif fill_missing == 'mode':
        for col in cleaned_df.columns:
            cleaned_df[col].fillna(cleaned_df[col].mode()[0] if not cleaned_df[col].mode().empty else None, inplace=True)
    
    return cleaned_df

def validate_dataset(df, required_columns=None):
    """
    Validate a DataFrame for required columns and data types.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of required column names. Default is None.
    
    Returns:
    tuple: (bool, str) indicating validation success and message.
    """
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    if df.empty:
        return False, "DataFrame is empty"
    
    return True, "Dataset validation passed"
import numpy as np

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a specified column using the Interquartile Range method.
    
    Args:
        data: numpy array or list containing the data
        column: index of the column to clean
    
    Returns:
        Cleaned data with outliers removed
    """
    if not isinstance(data, np.ndarray):
        data = np.array(data)
    
    column_data = data[:, column].astype(float)
    
    Q1 = np.percentile(column_data, 25)
    Q3 = np.percentile(column_data, 75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    mask = (column_data >= lower_bound) & (column_data <= upper_bound)
    
    return data[mask]

def calculate_statistics(data, column):
    """
    Calculate basic statistics for a column.
    
    Args:
        data: numpy array containing the data
        column: index of the column to analyze
    
    Returns:
        Dictionary containing mean, median, std, min, max
    """
    if not isinstance(data, np.ndarray):
        data = np.array(data)
    
    column_data = data[:, column].astype(float)
    
    stats = {
        'mean': np.mean(column_data),
        'median': np.median(column_data),
        'std': np.std(column_data),
        'min': np.min(column_data),
        'max': np.max(column_data)
    }
    
    return stats

def normalize_column(data, column, method='minmax'):
    """
    Normalize a column using specified method.
    
    Args:
        data: numpy array containing the data
        column: index of the column to normalize
        method: normalization method ('minmax' or 'zscore')
    
    Returns:
        Data with normalized column
    """
    if not isinstance(data, np.ndarray):
        data = np.array(data)
    
    column_data = data[:, column].astype(float).copy()
    
    if method == 'minmax':
        min_val = np.min(column_data)
        max_val = np.max(column_data)
        if max_val != min_val:
            column_data = (column_data - min_val) / (max_val - min_val)
    
    elif method == 'zscore':
        mean_val = np.mean(column_data)
        std_val = np.std(column_data)
        if std_val != 0:
            column_data = (column_data - mean_val) / std_val
    
    data[:, column] = column_data
    return data

def validate_data(data, expected_columns):
    """
    Validate data structure and dimensions.
    
    Args:
        data: numpy array to validate
        expected_columns: expected number of columns
    
    Returns:
        Boolean indicating if data is valid
    """
    if not isinstance(data, np.ndarray):
        return False
    
    if len(data.shape) != 2:
        return False
    
    if data.shape[1] != expected_columns:
        return False
    
    return True