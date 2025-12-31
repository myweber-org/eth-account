
import pandas as pd

def clean_dataset(df, columns_to_check=None):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        columns_to_check (list, optional): Specific columns to check for duplicates.
            If None, checks all columns. Defaults to None.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    # Create a copy to avoid modifying the original
    cleaned_df = df.copy()
    
    # Remove duplicate rows
    if columns_to_check:
        cleaned_df = cleaned_df.drop_duplicates(subset=columns_to_check)
    else:
        cleaned_df = cleaned_df.drop_duplicates()
    
    # Fill missing numeric values with column mean
    numeric_cols = cleaned_df.select_dtypes(include=['number']).columns
    for col in numeric_cols:
        if cleaned_df[col].isnull().any():
            cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mean())
    
    # Fill missing categorical values with mode
    categorical_cols = cleaned_df.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        if cleaned_df[col].isnull().any():
            mode_value = cleaned_df[col].mode()[0] if not cleaned_df[col].mode().empty else 'Unknown'
            cleaned_df[col] = cleaned_df[col].fillna(mode_value)
    
    # Reset index after cleaning
    cleaned_df = cleaned_df.reset_index(drop=True)
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate that a DataFrame meets basic requirements.
    
    Args:
        df (pd.DataFrame): DataFrame to validate.
        required_columns (list, optional): List of columns that must be present.
    
    Returns:
        tuple: (is_valid, error_message)
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
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(dataframe, column, multiplier=1.5):
    """
    Remove outliers from a DataFrame column using the IQR method.
    
    Args:
        dataframe: pandas DataFrame
        column: Column name to process
        multiplier: IQR multiplier (default 1.5)
    
    Returns:
        DataFrame with outliers removed
    """
    if column not in dataframe.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    q1 = dataframe[column].quantile(0.25)
    q3 = dataframe[column].quantile(0.75)
    iqr = q3 - q1
    
    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr
    
    filtered_df = dataframe[(dataframe[column] >= lower_bound) & 
                           (dataframe[column] <= upper_bound)]
    
    return filtered_df

def z_score_normalize(dataframe, columns):
    """
    Apply Z-score normalization to specified columns.
    
    Args:
        dataframe: pandas DataFrame
        columns: List of column names to normalize
    
    Returns:
        DataFrame with normalized columns
    """
    result_df = dataframe.copy()
    
    for col in columns:
        if col not in result_df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame")
        
        if result_df[col].dtype in [np.float64, np.int64]:
            mean_val = result_df[col].mean()
            std_val = result_df[col].std()
            
            if std_val > 0:
                result_df[col] = (result_df[col] - mean_val) / std_val
            else:
                result_df[col] = 0
    
    return result_df

def min_max_normalize(dataframe, columns, feature_range=(0, 1)):
    """
    Apply Min-Max normalization to specified columns.
    
    Args:
        dataframe: pandas DataFrame
        columns: List of column names to normalize
        feature_range: Tuple of (min, max) for output range
    
    Returns:
        DataFrame with normalized columns
    """
    result_df = dataframe.copy()
    min_val, max_val = feature_range
    
    for col in columns:
        if col not in result_df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame")
        
        if result_df[col].dtype in [np.float64, np.int64]:
            col_min = result_df[col].min()
            col_max = result_df[col].max()
            
            if col_max > col_min:
                result_df[col] = (result_df[col] - col_min) / (col_max - col_min)
                result_df[col] = result_df[col] * (max_val - min_val) + min_val
            else:
                result_df[col] = min_val
    
    return result_df

def detect_missing_patterns(dataframe, threshold=0.3):
    """
    Detect columns with high percentage of missing values.
    
    Args:
        dataframe: pandas DataFrame
        threshold: Missing value percentage threshold
    
    Returns:
        List of column names exceeding the threshold
    """
    missing_percentages = dataframe.isnull().sum() / len(dataframe)
    problematic_columns = missing_percentages[missing_percentages > threshold].index.tolist()
    
    return problematic_columns

def clean_dataset(dataframe, numeric_columns, outlier_multiplier=1.5, normalize_method='zscore'):
    """
    Comprehensive data cleaning pipeline.
    
    Args:
        dataframe: Input DataFrame
        numeric_columns: List of numeric columns to process
        outlier_multiplier: IQR multiplier for outlier removal
        normalize_method: 'zscore' or 'minmax' normalization
    
    Returns:
        Cleaned and normalized DataFrame
    """
    cleaned_df = dataframe.copy()
    
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, col, outlier_multiplier)
    
    if normalize_method == 'zscore':
        cleaned_df = z_score_normalize(cleaned_df, numeric_columns)
    elif normalize_method == 'minmax':
        cleaned_df = min_max_normalize(cleaned_df, numeric_columns)
    
    return cleaned_df