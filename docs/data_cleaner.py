
def remove_duplicates_preserve_order(seq):
    seen = set()
    result = []
    for item in seq:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range method.
    
    Parameters:
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

def calculate_summary_statistics(df):
    """
    Calculate summary statistics for numeric columns.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    
    Returns:
    pd.DataFrame: Summary statistics
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) == 0:
        return pd.DataFrame()
    
    stats = df[numeric_cols].agg(['mean', 'median', 'std', 'min', 'max'])
    
    return stats

def handle_missing_values(df, strategy='mean'):
    """
    Handle missing values in numeric columns.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    strategy (str): Imputation strategy ('mean', 'median', 'mode', or 'drop')
    
    Returns:
    pd.DataFrame: DataFrame with handled missing values
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if strategy == 'drop':
        return df.dropna(subset=numeric_cols)
    
    for col in numeric_cols:
        if df[col].isnull().any():
            if strategy == 'mean':
                fill_value = df[col].mean()
            elif strategy == 'median':
                fill_value = df[col].median()
            elif strategy == 'mode':
                fill_value = df[col].mode()[0]
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
            
            df[col] = df[col].fillna(fill_value)
    
    return df

def normalize_data(df, columns=None):
    """
    Normalize specified columns using min-max scaling.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    columns (list): List of column names to normalize
    
    Returns:
    pd.DataFrame: DataFrame with normalized columns
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    result_df = df.copy()
    
    for col in columns:
        if col in df.columns and np.issubdtype(df[col].dtype, np.number):
            min_val = df[col].min()
            max_val = df[col].max()
            
            if max_val > min_val:
                result_df[col] = (df[col] - min_val) / (max_val - min_val)
    
    return result_df

def clean_dataset(df, outlier_columns=None, missing_strategy='mean'):
    """
    Comprehensive data cleaning pipeline.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    outlier_columns (list): Columns to apply outlier removal
    missing_strategy (str): Strategy for handling missing values
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    cleaned_df = handle_missing_values(cleaned_df, strategy=missing_strategy)
    
    if outlier_columns:
        for col in outlier_columns:
            if col in cleaned_df.columns:
                cleaned_df = remove_outliers_iqr(cleaned_df, col)
    
    return cleaned_dfimport numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column, threshold=1.5):
    """
    Remove outliers using IQR method.
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr
    
    return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)].copy()

def remove_outliers_zscore(data, column, threshold=3):
    """
    Remove outliers using Z-score method.
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    z_scores = np.abs(stats.zscore(data[column].dropna()))
    mask = z_scores < threshold
    valid_indices = data[column].dropna().index[mask]
    
    return data.loc[valid_indices].copy()

def normalize_minmax(data, column):
    """
    Normalize column using Min-Max scaling.
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    min_val = data[column].min()
    max_val = data[column].max()
    
    if max_val == min_val:
        return data[column].copy()
    
    normalized = (data[column] - min_val) / (max_val - min_val)
    return normalized

def normalize_zscore(data, column):
    """
    Normalize column using Z-score standardization.
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    mean_val = data[column].mean()
    std_val = data[column].std()
    
    if std_val == 0:
        return data[column].copy()
    
    standardized = (data[column] - mean_val) / std_val
    return standardized

def clean_dataset(df, numeric_columns=None, outlier_method='iqr', normalize_method='minmax'):
    """
    Comprehensive data cleaning pipeline.
    """
    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_df = df.copy()
    
    for col in numeric_columns:
        if col not in cleaned_df.columns:
            continue
            
        if outlier_method == 'iqr':
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
        elif outlier_method == 'zscore':
            cleaned_df = remove_outliers_zscore(cleaned_df, col)
        
        if normalize_method == 'minmax':
            cleaned_df[col] = normalize_minmax(cleaned_df, col)
        elif normalize_method == 'zscore':
            cleaned_df[col] = normalize_zscore(cleaned_df, col)
    
    return cleaned_df.reset_index(drop=True)

def validate_data(df, required_columns=None, allow_nan_ratio=0.1):
    """
    Validate data quality.
    """
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
    
    nan_ratio = df.isnull().sum().sum() / (df.shape[0] * df.shape[1])
    if nan_ratio > allow_nan_ratio:
        raise ValueError(f"NaN ratio {nan_ratio:.2%} exceeds allowed threshold {allow_nan_ratio:.2%}")
    
    return True
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column, factor=1.5):
    """
    Remove outliers using Interquartile Range method
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - factor * iqr
    upper_bound = q3 + factor * iqr
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data.copy()

def z_score_normalization(data, column):
    """
    Apply Z-score normalization to specified column
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    normalized_data = data.copy()
    mean_val = normalized_data[column].mean()
    std_val = normalized_data[column].std()
    
    if std_val > 0:
        normalized_data[f'{column}_normalized'] = (normalized_data[column] - mean_val) / std_val
    else:
        normalized_data[f'{column}_normalized'] = 0
    
    return normalized_data

def min_max_scaling(data, column, feature_range=(0, 1)):
    """
    Apply Min-Max scaling to specified column
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    scaled_data = data.copy()
    min_val = scaled_data[column].min()
    max_val = scaled_data[column].max()
    
    if max_val > min_val:
        scaled_data[f'{column}_scaled'] = (scaled_data[column] - min_val) / (max_val - min_val)
        scaled_data[f'{column}_scaled'] = scaled_data[f'{column}_scaled'] * (feature_range[1] - feature_range[0]) + feature_range[0]
    else:
        scaled_data[f'{column}_scaled'] = feature_range[0]
    
    return scaled_data

def handle_missing_values(data, strategy='mean', columns=None):
    """
    Handle missing values in specified columns
    """
    processed_data = data.copy()
    
    if columns is None:
        columns = processed_data.columns
    
    for column in columns:
        if column in processed_data.columns and processed_data[column].isnull().any():
            if strategy == 'mean':
                fill_value = processed_data[column].mean()
            elif strategy == 'median':
                fill_value = processed_data[column].median()
            elif strategy == 'mode':
                fill_value = processed_data[column].mode()[0]
            elif strategy == 'constant':
                fill_value = 0
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
            
            processed_data[column] = processed_data[column].fillna(fill_value)
    
    return processed_data

def validate_dataframe(data):
    """
    Basic DataFrame validation
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    
    if data.empty:
        raise ValueError("DataFrame is empty")
    
    return True

def clean_dataset(data, numeric_columns=None, outlier_factor=1.5, normalization_method='zscore'):
    """
    Comprehensive data cleaning pipeline
    """
    validate_dataframe(data)
    
    cleaned_data = data.copy()
    
    if numeric_columns is None:
        numeric_columns = cleaned_data.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_data = handle_missing_values(cleaned_data, strategy='mean', columns=numeric_columns)
    
    for column in numeric_columns:
        if column in cleaned_data.columns:
            cleaned_data = remove_outliers_iqr(cleaned_data, column, factor=outlier_factor)
            
            if normalization_method == 'zscore':
                cleaned_data = z_score_normalization(cleaned_data, column)
            elif normalization_method == 'minmax':
                cleaned_data = min_max_scaling(cleaned_data, column)
    
    return cleaned_data