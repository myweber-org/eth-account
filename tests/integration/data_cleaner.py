
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column, factor=1.5):
    """
    Remove outliers using IQR method
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - factor * IQR
    upper_bound = Q3 + factor * IQR
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    removed_count = len(data) - len(filtered_data)
    
    return filtered_data, removed_count

def remove_outliers_zscore(data, column, threshold=3):
    """
    Remove outliers using Z-score method
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    z_scores = np.abs(stats.zscore(data[column]))
    filtered_data = data[z_scores < threshold]
    removed_count = len(data) - len(filtered_data)
    
    return filtered_data, removed_count

def normalize_minmax(data, column):
    """
    Normalize data using Min-Max scaling
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    min_val = data[column].min()
    max_val = data[column].max()
    
    if max_val == min_val:
        return data[column].apply(lambda x: 0.5)
    
    normalized = (data[column] - min_val) / (max_val - min_val)
    return normalized

def normalize_zscore(data, column):
    """
    Normalize data using Z-score standardization
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    mean_val = data[column].mean()
    std_val = data[column].std()
    
    if std_val == 0:
        return data[column].apply(lambda x: 0)
    
    standardized = (data[column] - mean_val) / std_val
    return standardized

def clean_dataset(data, numeric_columns=None, outlier_method='iqr', normalize_method='zscore'):
    """
    Main function to clean dataset by removing outliers and normalizing numeric columns
    """
    if numeric_columns is None:
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_data = data.copy()
    report = {
        'original_rows': len(data),
        'outliers_removed': {},
        'columns_normalized': []
    }
    
    for column in numeric_columns:
        if column not in cleaned_data.columns:
            continue
            
        if outlier_method == 'iqr':
            cleaned_data, removed = remove_outliers_iqr(cleaned_data, column)
        elif outlier_method == 'zscore':
            cleaned_data, removed = remove_outliers_zscore(cleaned_data, column)
        else:
            raise ValueError(f"Unknown outlier method: {outlier_method}")
        
        report['outliers_removed'][column] = removed
        
        if normalize_method == 'minmax':
            cleaned_data[column] = normalize_minmax(cleaned_data, column)
        elif normalize_method == 'zscore':
            cleaned_data[column] = normalize_zscore(cleaned_data, column)
        else:
            raise ValueError(f"Unknown normalize method: {normalize_method}")
        
        report['columns_normalized'].append(column)
    
    report['final_rows'] = len(cleaned_data)
    report['rows_removed'] = report['original_rows'] - report['final_rows']
    
    return cleaned_data, report

def validate_data(data, required_columns=None, allow_nan=False):
    """
    Validate dataset structure and content
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
    
    if not allow_nan and data.isnull().any().any():
        nan_columns = data.columns[data.isnull().any()].tolist()
        raise ValueError(f"NaN values found in columns: {nan_columns}")
    
    return True
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(dataframe, column, threshold=1.5):
    """
    Remove outliers from a DataFrame column using IQR method.
    
    Args:
        dataframe: pandas DataFrame
        column: column name to process
        threshold: IQR multiplier (default 1.5)
    
    Returns:
        DataFrame with outliers removed
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
    
    return filtered_df

def normalize_minmax(dataframe, columns=None):
    """
    Normalize specified columns using min-max scaling.
    
    Args:
        dataframe: pandas DataFrame
        columns: list of column names to normalize (default: all numeric columns)
    
    Returns:
        DataFrame with normalized columns
    """
    if columns is None:
        columns = dataframe.select_dtypes(include=[np.number]).columns.tolist()
    
    normalized_df = dataframe.copy()
    
    for col in columns:
        if col in dataframe.columns and pd.api.types.is_numeric_dtype(dataframe[col]):
            col_min = normalized_df[col].min()
            col_max = normalized_df[col].max()
            
            if col_max != col_min:
                normalized_df[col] = (normalized_df[col] - col_min) / (col_max - col_min)
            else:
                normalized_df[col] = 0
    
    return normalized_df

def detect_skewed_columns(dataframe, skew_threshold=0.5):
    """
    Identify columns with significant skewness.
    
    Args:
        dataframe: pandas DataFrame
        skew_threshold: absolute skewness threshold (default 0.5)
    
    Returns:
        Dictionary with skewed columns and their skewness values
    """
    numeric_cols = dataframe.select_dtypes(include=[np.number]).columns
    skewed_cols = {}
    
    for col in numeric_cols:
        skewness = dataframe[col].skew()
        if abs(skewness) > skew_threshold:
            skewed_cols[col] = skewness
    
    return skewed_cols

def log_transform_skewed(dataframe, skewed_columns):
    """
    Apply log transformation to reduce skewness.
    
    Args:
        dataframe: pandas DataFrame
        skewed_columns: list of column names to transform
    
    Returns:
        DataFrame with transformed columns
    """
    transformed_df = dataframe.copy()
    
    for col in skewed_columns:
        if col in transformed_df.columns:
            min_val = transformed_df[col].min()
            if min_val <= 0:
                transformed_df[col] = np.log1p(transformed_df[col] - min_val + 1)
            else:
                transformed_df[col] = np.log(transformed_df[col])
    
    return transformed_df

def clean_dataset(dataframe, outlier_columns=None, normalize=True, handle_skewness=True):
    """
    Comprehensive data cleaning pipeline.
    
    Args:
        dataframe: pandas DataFrame to clean
        outlier_columns: columns to apply outlier removal (default: all numeric)
        normalize: whether to apply min-max normalization
        handle_skewness: whether to handle skewed columns
    
    Returns:
        Cleaned DataFrame
    """
    cleaned_df = dataframe.copy()
    
    if outlier_columns is None:
        outlier_columns = cleaned_df.select_dtypes(include=[np.number]).columns.tolist()
    
    for col in outlier_columns:
        if col in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
    
    if handle_skewness:
        skewed_cols = detect_skewed_columns(cleaned_df)
        if skewed_cols:
            cleaned_df = log_transform_skewed(cleaned_df, list(skewed_cols.keys()))
    
    if normalize:
        cleaned_df = normalize_minmax(cleaned_df)
    
    return cleaned_df