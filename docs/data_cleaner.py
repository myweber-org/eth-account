import pandas as pd
import numpy as np

def clean_missing_data(df, strategy='mean', columns=None):
    """
    Clean missing values in a DataFrame using specified strategy.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        strategy (str): Strategy for handling missing values.
                        Options: 'mean', 'median', 'mode', 'drop', 'fill_zero'
        columns (list): List of columns to apply cleaning to. If None, applies to all numeric columns.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    df_cleaned = df.copy()
    
    if columns is None:
        numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
        columns = list(numeric_cols)
    
    if strategy == 'drop':
        df_cleaned = df_cleaned.dropna(subset=columns)
    elif strategy == 'mean':
        for col in columns:
            if col in df_cleaned.columns:
                df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].mean())
    elif strategy == 'median':
        for col in columns:
            if col in df_cleaned.columns:
                df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].median())
    elif strategy == 'mode':
        for col in columns:
            if col in df_cleaned.columns:
                df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].mode()[0])
    elif strategy == 'fill_zero':
        for col in columns:
            if col in df_cleaned.columns:
                df_cleaned[col] = df_cleaned[col].fillna(0)
    
    return df_cleaned

def detect_outliers_iqr(df, columns=None, threshold=1.5):
    """
    Detect outliers using Interquartile Range method.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        columns (list): List of columns to check for outliers
        threshold (float): IQR multiplier threshold
    
    Returns:
        dict: Dictionary with outlier counts per column
    """
    if columns is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        columns = list(numeric_cols)
    
    outliers = {}
    
    for col in columns:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
            outliers[col] = outlier_mask.sum()
    
    return outliers

def normalize_data(df, columns=None, method='minmax'):
    """
    Normalize data in specified columns.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        columns (list): List of columns to normalize
        method (str): Normalization method ('minmax' or 'zscore')
    
    Returns:
        pd.DataFrame: DataFrame with normalized columns
    """
    df_normalized = df.copy()
    
    if columns is None:
        numeric_cols = df_normalized.select_dtypes(include=[np.number]).columns
        columns = list(numeric_cols)
    
    for col in columns:
        if col in df_normalized.columns:
            if method == 'minmax':
                min_val = df_normalized[col].min()
                max_val = df_normalized[col].max()
                if max_val != min_val:
                    df_normalized[col] = (df_normalized[col] - min_val) / (max_val - min_val)
            elif method == 'zscore':
                mean_val = df_normalized[col].mean()
                std_val = df_normalized[col].std()
                if std_val != 0:
                    df_normalized[col] = (df_normalized[col] - mean_val) / std_val
    
    return df_normalized

def validate_dataframe(df, required_columns=None, min_rows=1):
    """
    Validate DataFrame structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        required_columns (list): List of required column names
        min_rows (int): Minimum number of rows required
    
    Returns:
        tuple: (is_valid, error_message)
    """
    if not isinstance(df, pd.DataFrame):
        return False, "Input is not a pandas DataFrame"
    
    if len(df) < min_rows:
        return False, f"DataFrame has fewer than {min_rows} rows"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    return True, "DataFrame is valid"

def save_cleaned_data(df, output_path, format='csv'):
    """
    Save cleaned DataFrame to file.
    
    Args:
        df (pd.DataFrame): DataFrame to save
        output_path (str): Path to save the file
        format (str): File format ('csv', 'excel', 'json')
    
    Returns:
        bool: True if save successful, False otherwise
    """
    try:
        if format == 'csv':
            df.to_csv(output_path, index=False)
        elif format == 'excel':
            df.to_excel(output_path, index=False)
        elif format == 'json':
            df.to_json(output_path, orient='records')
        else:
            return False
        
        return True
    except Exception as e:
        print(f"Error saving file: {e}")
        return False
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(dataframe, column, threshold=1.5):
    """
    Remove outliers using IQR method
    """
    if column not in dataframe.columns:
        raise ValueError(f"Column '{column}' not found in dataframe")
    
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
    Normalize data using min-max scaling
    """
    if columns is None:
        columns = dataframe.select_dtypes(include=[np.number]).columns
    
    normalized_df = dataframe.copy()
    
    for col in columns:
        if col in dataframe.columns and pd.api.types.is_numeric_dtype(dataframe[col]):
            min_val = dataframe[col].min()
            max_val = dataframe[col].max()
            
            if max_val != min_val:
                normalized_df[col] = (dataframe[col] - min_val) / (max_val - min_val)
            else:
                normalized_df[col] = 0
    
    return normalized_df

def standardize_zscore(dataframe, columns=None):
    """
    Standardize data using z-score normalization
    """
    if columns is None:
        columns = dataframe.select_dtypes(include=[np.number]).columns
    
    standardized_df = dataframe.copy()
    
    for col in columns:
        if col in dataframe.columns and pd.api.types.is_numeric_dtype(dataframe[col]):
            mean_val = dataframe[col].mean()
            std_val = dataframe[col].std()
            
            if std_val > 0:
                standardized_df[col] = (dataframe[col] - mean_val) / std_val
            else:
                standardized_df[col] = 0
    
    return standardized_df

def handle_missing_values(dataframe, strategy='mean', columns=None):
    """
    Handle missing values with different strategies
    """
    if columns is None:
        columns = dataframe.select_dtypes(include=[np.number]).columns
    
    processed_df = dataframe.copy()
    
    for col in columns:
        if col in dataframe.columns and pd.api.types.is_numeric_dtype(dataframe[col]):
            if strategy == 'mean':
                fill_value = dataframe[col].mean()
            elif strategy == 'median':
                fill_value = dataframe[col].median()
            elif strategy == 'mode':
                fill_value = dataframe[col].mode()[0] if not dataframe[col].mode().empty else 0
            elif strategy == 'zero':
                fill_value = 0
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
            
            processed_df[col] = dataframe[col].fillna(fill_value)
    
    return processed_df

def clean_dataset(dataframe, config=None):
    """
    Comprehensive data cleaning pipeline
    """
    if config is None:
        config = {
            'outlier_removal': False,
            'normalization': 'none',
            'missing_values': 'mean'
        }
    
    df_clean = dataframe.copy()
    
    if config.get('missing_values') != 'none':
        df_clean = handle_missing_values(df_clean, strategy=config['missing_values'])
    
    if config.get('outlier_removal'):
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df_clean = remove_outliers_iqr(df_clean, col)
    
    if config.get('normalization') == 'minmax':
        df_clean = normalize_minmax(df_clean)
    elif config.get('normalization') == 'zscore':
        df_clean = standardize_zscore(df_clean)
    
    return df_clean