
import pandas as pd

def remove_duplicates(df, subset=None, keep='first'):
    """
    Remove duplicate rows from a DataFrame.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    subset (list, optional): Column labels to consider for duplicates
    keep (str, optional): Which duplicates to keep ('first', 'last', False)
    
    Returns:
    pd.DataFrame: DataFrame with duplicates removed
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    
    cleaned_df = df.drop_duplicates(subset=subset, keep=keep)
    
    removed_count = len(df) - len(cleaned_df)
    print(f"Removed {removed_count} duplicate rows")
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and required columns.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate
    required_columns (list, optional): List of required column names
    
    Returns:
    bool: True if validation passes, False otherwise
    """
    if not isinstance(df, pd.DataFrame):
        print("Error: Input is not a DataFrame")
        return False
    
    if df.empty:
        print("Warning: DataFrame is empty")
        return True
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Error: Missing required columns: {missing_columns}")
            return False
    
    return True

def clean_numeric_column(df, column_name, fill_method='mean'):
    """
    Clean numeric column by handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column_name (str): Name of column to clean
    fill_method (str): Method to fill missing values ('mean', 'median', 'zero')
    
    Returns:
    pd.DataFrame: DataFrame with cleaned column
    """
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame")
    
    if not pd.api.types.is_numeric_dtype(df[column_name]):
        raise TypeError(f"Column '{column_name}' must be numeric")
    
    df_clean = df.copy()
    
    missing_count = df_clean[column_name].isna().sum()
    if missing_count > 0:
        if fill_method == 'mean':
            fill_value = df_clean[column_name].mean()
        elif fill_method == 'median':
            fill_value = df_clean[column_name].median()
        elif fill_method == 'zero':
            fill_value = 0
        else:
            raise ValueError("fill_method must be 'mean', 'median', or 'zero'")
        
        df_clean[column_name] = df_clean[column_name].fillna(fill_value)
        print(f"Filled {missing_count} missing values with {fill_method}: {fill_value}")
    
    return df_clean
import numpy as np
import pandas as pd

def remove_outliers_iqr(dataframe, column):
    Q1 = dataframe[column].quantile(0.25)
    Q3 = dataframe[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    filtered_df = dataframe[(dataframe[column] >= lower_bound) & (dataframe[column] <= upper_bound)]
    return filtered_df

def normalize_column(dataframe, column, method='minmax'):
    if method == 'minmax':
        min_val = dataframe[column].min()
        max_val = dataframe[column].max()
        if max_val - min_val != 0:
            dataframe[column] = (dataframe[column] - min_val) / (max_val - min_val)
    elif method == 'zscore':
        mean_val = dataframe[column].mean()
        std_val = dataframe[column].std()
        if std_val != 0:
            dataframe[column] = (dataframe[column] - mean_val) / std_val
    return dataframe

def clean_dataset(dataframe, numeric_columns):
    cleaned_df = dataframe.copy()
    for column in numeric_columns:
        if column in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, column)
            cleaned_df = normalize_column(cleaned_df, column, method='zscore')
    cleaned_df = cleaned_df.dropna()
    return cleaned_df.reset_index(drop=True)

def generate_summary(dataframe):
    summary = {}
    for column in dataframe.select_dtypes(include=[np.number]).columns:
        summary[column] = {
            'mean': dataframe[column].mean(),
            'median': dataframe[column].median(),
            'std': dataframe[column].std(),
            'min': dataframe[column].min(),
            'max': dataframe[column].max()
        }
    return summary
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(dataframe, column, threshold=1.5):
    """
    Remove outliers from a DataFrame column using IQR method.
    
    Parameters:
    dataframe (pd.DataFrame): Input DataFrame
    column (str): Column name to process
    threshold (float): IQR multiplier (default: 1.5)
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed
    """
    if column not in dataframe.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    q1 = dataframe[column].quantile(0.25)
    q3 = dataframe[column].quantile(0.75)
    iqr = q3 - q1
    
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr
    
    filtered_df = dataframe[(dataframe[column] >= lower_bound) & 
                           (dataframe[column] <= upper_bound)]
    
    return filtered_df.copy()

def normalize_column(dataframe, column, method='zscore'):
    """
    Normalize a column using specified method.
    
    Parameters:
    dataframe (pd.DataFrame): Input DataFrame
    column (str): Column name to normalize
    method (str): Normalization method ('zscore', 'minmax', 'robust')
    
    Returns:
    pd.DataFrame: DataFrame with normalized column
    """
    if column not in dataframe.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    df_copy = dataframe.copy()
    
    if method == 'zscore':
        mean_val = df_copy[column].mean()
        std_val = df_copy[column].std()
        if std_val > 0:
            df_copy[f'{column}_normalized'] = (df_copy[column] - mean_val) / std_val
        else:
            df_copy[f'{column}_normalized'] = 0
    
    elif method == 'minmax':
        min_val = df_copy[column].min()
        max_val = df_copy[column].max()
        range_val = max_val - min_val
        if range_val > 0:
            df_copy[f'{column}_normalized'] = (df_copy[column] - min_val) / range_val
        else:
            df_copy[f'{column}_normalized'] = 0
    
    elif method == 'robust':
        median_val = df_copy[column].median()
        iqr_val = stats.iqr(df_copy[column])
        if iqr_val > 0:
            df_copy[f'{column}_normalized'] = (df_copy[column] - median_val) / iqr_val
        else:
            df_copy[f'{column}_normalized'] = 0
    
    else:
        raise ValueError("Method must be 'zscore', 'minmax', or 'robust'")
    
    return df_copy

def clean_dataset(dataframe, numeric_columns=None, outlier_threshold=1.5, 
                  normalize_method='zscore', drop_na=True):
    """
    Comprehensive dataset cleaning function.
    
    Parameters:
    dataframe (pd.DataFrame): Input DataFrame
    numeric_columns (list): List of numeric columns to process
    outlier_threshold (float): IQR threshold for outlier removal
    normalize_method (str): Normalization method
    drop_na (bool): Whether to drop rows with missing values
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    if numeric_columns is None:
        numeric_columns = dataframe.select_dtypes(include=[np.number]).columns.tolist()
    
    df_clean = dataframe.copy()
    
    if drop_na:
        df_clean = df_clean.dropna()
    
    for column in numeric_columns:
        if column in df_clean.columns:
            df_clean = remove_outliers_iqr(df_clean, column, outlier_threshold)
            df_clean = normalize_column(df_clean, column, normalize_method)
    
    return df_clean.reset_index(drop=True)

def validate_dataframe(dataframe, required_columns=None, min_rows=1):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    dataframe (pd.DataFrame): DataFrame to validate
    required_columns (list): List of required column names
    min_rows (int): Minimum number of rows required
    
    Returns:
    tuple: (is_valid, error_message)
    """
    if not isinstance(dataframe, pd.DataFrame):
        return False, "Input is not a pandas DataFrame"
    
    if len(dataframe) < min_rows:
        return False, f"DataFrame must have at least {min_rows} rows"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in dataframe.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    return True, "DataFrame is valid"