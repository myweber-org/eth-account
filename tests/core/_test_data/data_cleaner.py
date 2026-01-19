
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
    
    result_df = dataframe.copy()
    
    for col in columns:
        if col in dataframe.columns and np.issubdtype(dataframe[col].dtype, np.number):
            col_min = dataframe[col].min()
            col_max = dataframe[col].max()
            
            if col_max != col_min:
                result_df[col] = (dataframe[col] - col_min) / (col_max - col_min)
            else:
                result_df[col] = 0
    
    return result_df

def detect_anomalies_zscore(dataframe, column, threshold=3):
    """
    Detect anomalies using Z-score method.
    
    Args:
        dataframe: pandas DataFrame
        column: column name to analyze
        threshold: Z-score threshold (default 3)
    
    Returns:
        Boolean Series indicating anomalies
    """
    if column not in dataframe.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    z_scores = np.abs(stats.zscore(dataframe[column].dropna()))
    anomalies = z_scores > threshold
    
    result_series = pd.Series(False, index=dataframe.index)
    valid_indices = dataframe[column].dropna().index
    result_series.loc[valid_indices] = anomalies
    
    return result_series

def clean_missing_values(dataframe, strategy='mean', columns=None):
    """
    Handle missing values in DataFrame.
    
    Args:
        dataframe: pandas DataFrame
        strategy: imputation strategy ('mean', 'median', 'mode', 'drop')
        columns: list of columns to process (default: all columns)
    
    Returns:
        DataFrame with handled missing values
    """
    if columns is None:
        columns = dataframe.columns.tolist()
    
    result_df = dataframe.copy()
    
    for col in columns:
        if col not in result_df.columns:
            continue
            
        if strategy == 'drop':
            result_df = result_df.dropna(subset=[col])
        elif strategy == 'mean' and np.issubdtype(result_df[col].dtype, np.number):
            result_df[col] = result_df[col].fillna(result_df[col].mean())
        elif strategy == 'median' and np.issubdtype(result_df[col].dtype, np.number):
            result_df[col] = result_df[col].fillna(result_df[col].median())
        elif strategy == 'mode':
            result_df[col] = result_df[col].fillna(result_df[col].mode()[0])
    
    return result_df

def validate_dataframe(dataframe, required_columns=None, min_rows=1):
    """
    Validate DataFrame structure and content.
    
    Args:
        dataframe: pandas DataFrame to validate
        required_columns: list of required column names
        min_rows: minimum number of rows required
    
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
    
    return True, "DataFrame validation passed"

def process_data_pipeline(dataframe, config):
    """
    Execute a data processing pipeline based on configuration.
    
    Args:
        dataframe: input pandas DataFrame
        config: dictionary with processing steps configuration
    
    Returns:
        Processed DataFrame
    """
    result_df = dataframe.copy()
    
    if 'missing_values' in config:
        strategy = config['missing_values'].get('strategy', 'mean')
        columns = config['missing_values'].get('columns')
        result_df = clean_missing_values(result_df, strategy, columns)
    
    if 'outliers' in config:
        for col in config['outliers'].get('columns', []):
            threshold = config['outliers'].get('threshold', 1.5)
            result_df = remove_outliers_iqr(result_df, col, threshold)
    
    if 'normalization' in config:
        columns = config['normalization'].get('columns')
        result_df = normalize_minmax(result_df, columns)
    
    return result_df