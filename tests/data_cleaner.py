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

def normalize_minmax(dataframe, column):
    min_val = dataframe[column].min()
    max_val = dataframe[column].max()
    if max_val - min_val == 0:
        return dataframe[column].apply(lambda x: 0.0)
    normalized = (dataframe[column] - min_val) / (max_val - min_val)
    return normalized

def clean_dataset(dataframe, numeric_columns):
    cleaned_df = dataframe.copy()
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
            cleaned_df[col] = normalize_minmax(cleaned_df, col)
    return cleaned_df

def calculate_statistics(dataframe, column):
    stats = {
        'mean': dataframe[column].mean(),
        'median': dataframe[column].median(),
        'std': dataframe[column].std(),
        'count': dataframe[column].count()
    }
    return stats
def remove_duplicates_preserve_order(sequence):
    seen = set()
    result = []
    for item in sequence:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return resultimport pandas as pd
import numpy as np

def remove_missing_rows(df, threshold=0.8):
    """
    Remove rows with missing values exceeding the threshold.
    
    Args:
        df: pandas DataFrame
        threshold: float, maximum allowed proportion of missing values per row
    
    Returns:
        Cleaned DataFrame
    """
    missing_prop = df.isnull().mean(axis=1)
    return df[missing_prop <= threshold].reset_index(drop=True)

def fill_missing_with_median(df, columns=None):
    """
    Fill missing values with column median.
    
    Args:
        df: pandas DataFrame
        columns: list of column names to process, if None processes all numeric columns
    
    Returns:
        DataFrame with filled values
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    df_filled = df.copy()
    for col in columns:
        if col in df.columns:
            median_val = df[col].median()
            df_filled[col].fillna(median_val, inplace=True)
    
    return df_filled

def remove_outliers_iqr(df, column, multiplier=1.5):
    """
    Remove outliers using the Interquartile Range method.
    
    Args:
        df: pandas DataFrame
        column: string, column name to check for outliers
        multiplier: float, IQR multiplier for outlier detection
    
    Returns:
        DataFrame without outliers in the specified column
    """
    if column not in df.columns:
        return df
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)].reset_index(drop=True)

def standardize_columns(df, columns=None):
    """
    Standardize specified columns to have zero mean and unit variance.
    
    Args:
        df: pandas DataFrame
        columns: list of column names to standardize, if None processes all numeric columns
    
    Returns:
        DataFrame with standardized columns
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    df_standardized = df.copy()
    for col in columns:
        if col in df.columns and df[col].std() > 0:
            df_standardized[col] = (df[col] - df[col].mean()) / df[col].std()
    
    return df_standardized

def clean_dataset(df, missing_threshold=0.8, outlier_columns=None, outlier_multiplier=1.5):
    """
    Comprehensive data cleaning pipeline.
    
    Args:
        df: pandas DataFrame
        missing_threshold: float for missing value removal
        outlier_columns: list of columns to check for outliers
        outlier_multiplier: float for IQR outlier detection
    
    Returns:
        Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    cleaned_df = remove_missing_rows(cleaned_df, threshold=missing_threshold)
    cleaned_df = fill_missing_with_median(cleaned_df)
    
    if outlier_columns:
        for col in outlier_columns:
            if col in cleaned_df.columns:
                cleaned_df = remove_outliers_iqr(cleaned_df, col, multiplier=outlier_multiplier)
    
    return cleaned_df.reset_index(drop=True)