import pandas as pd

def clean_dataframe(df, drop_duplicates=True, fill_missing=False, fill_value=0):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    drop_duplicates (bool): If True, remove duplicate rows.
    fill_missing (bool): If True, fill missing values with fill_value.
    fill_value: Value to use for filling missing values.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    if fill_missing:
        cleaned_df = cleaned_df.fillna(fill_value)
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of column names that must be present.
    
    Returns:
    tuple: (is_valid, error_message)
    """
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
    
    if df.empty:
        return False, "DataFrame is empty"
    
    return True, "DataFrame is valid"

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'A': [1, 2, 2, None, 5],
        'B': [10, 20, 20, 40, None],
        'C': ['x', 'y', 'y', 'z', 'z']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nCleaned DataFrame:")
    cleaned = clean_dataframe(df, drop_duplicates=True, fill_missing=True, fill_value=0)
    print(cleaned)
    
    is_valid, message = validate_dataframe(cleaned, required_columns=['A', 'B', 'C'])
    print(f"\nValidation: {message}")
import pandas as pd
import numpy as np
from scipy import stats

def normalize_data(df, columns=None, method='zscore'):
    """
    Normalize specified columns in DataFrame.
    
    Args:
        df: pandas DataFrame
        columns: list of column names to normalize (default: all numeric columns)
        method: normalization method ('zscore', 'minmax', or 'robust')
    
    Returns:
        Normalized DataFrame
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    df_normalized = df.copy()
    
    for col in columns:
        if col not in df.columns:
            continue
            
        if method == 'zscore':
            df_normalized[col] = stats.zscore(df[col])
        elif method == 'minmax':
            col_min = df[col].min()
            col_max = df[col].max()
            if col_max != col_min:
                df_normalized[col] = (df[col] - col_min) / (col_max - col_min)
        elif method == 'robust':
            median = df[col].median()
            iqr = df[col].quantile(0.75) - df[col].quantile(0.25)
            if iqr != 0:
                df_normalized[col] = (df[col] - median) / iqr
    
    return df_normalized

def remove_outliers(df, columns=None, method='iqr', threshold=1.5):
    """
    Remove outliers from specified columns.
    
    Args:
        df: pandas DataFrame
        columns: list of column names to process (default: all numeric columns)
        method: outlier detection method ('iqr' or 'zscore')
        threshold: threshold value for outlier detection
    
    Returns:
        DataFrame with outliers removed
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    df_clean = df.copy()
    
    for col in columns:
        if col not in df.columns:
            continue
            
        if method == 'iqr':
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            mask = (df[col] >= lower_bound) & (df[col] <= upper_bound)
        elif method == 'zscore':
            z_scores = np.abs(stats.zscore(df[col].fillna(df[col].mean())))
            mask = z_scores < threshold
        
        df_clean = df_clean[mask]
    
    return df_clean.reset_index(drop=True)

def handle_missing_values(df, columns=None, strategy='mean'):
    """
    Handle missing values in specified columns.
    
    Args:
        df: pandas DataFrame
        columns: list of column names to process (default: all columns)
        strategy: imputation strategy ('mean', 'median', 'mode', or 'drop')
    
    Returns:
        DataFrame with handled missing values
    """
    if columns is None:
        columns = df.columns.tolist()
    
    df_processed = df.copy()
    
    for col in columns:
        if col not in df.columns:
            continue
            
        if strategy == 'mean' and pd.api.types.is_numeric_dtype(df[col]):
            df_processed[col] = df[col].fillna(df[col].mean())
        elif strategy == 'median' and pd.api.types.is_numeric_dtype(df[col]):
            df_processed[col] = df[col].fillna(df[col].median())
        elif strategy == 'mode':
            df_processed[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else None)
        elif strategy == 'drop':
            df_processed = df_processed.dropna(subset=[col])
    
    return df_processed.reset_index(drop=True) if strategy == 'drop' else df_processed

def clean_dataset(df, config=None):
    """
    Comprehensive data cleaning pipeline.
    
    Args:
        df: pandas DataFrame
        config: dictionary with cleaning configuration
    
    Returns:
        Cleaned DataFrame
    """
    if config is None:
        config = {
            'missing_values': {'strategy': 'mean'},
            'outliers': {'method': 'iqr', 'threshold': 1.5},
            'normalization': {'method': 'zscore'}
        }
    
    df_clean = df.copy()
    
    # Handle missing values
    df_clean = handle_missing_values(
        df_clean, 
        strategy=config.get('missing_values', {}).get('strategy', 'mean')
    )
    
    # Remove outliers
    df_clean = remove_outliers(
        df_clean,
        method=config.get('outliers', {}).get('method', 'iqr'),
        threshold=config.get('outliers', {}).get('threshold', 1.5)
    )
    
    # Normalize data
    df_clean = normalize_data(
        df_clean,
        method=config.get('normalization', {}).get('method', 'zscore')
    )
    
    return df_clean

def get_cleaning_report(df, df_clean):
    """
    Generate a report comparing original and cleaned datasets.
    
    Args:
        df: original DataFrame
        df_clean: cleaned DataFrame
    
    Returns:
        Dictionary with cleaning statistics
    """
    report = {
        'original_rows': len(df),
        'cleaned_rows': len(df_clean),
        'rows_removed': len(df) - len(df_clean),
        'removal_percentage': ((len(df) - len(df_clean)) / len(df)) * 100 if len(df) > 0 else 0,
        'original_columns': len(df.columns),
        'cleaned_columns': len(df_clean.columns),
        'missing_values_original': df.isnull().sum().sum(),
        'missing_values_cleaned': df_clean.isnull().sum().sum()
    }
    
    return report