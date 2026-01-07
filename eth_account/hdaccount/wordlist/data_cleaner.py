
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column, factor=1.5):
    """
    Remove outliers using the Interquartile Range method.
    
    Args:
        data: pandas DataFrame
        column: column name to process
        factor: IQR multiplier (default 1.5)
    
    Returns:
        DataFrame with outliers removed
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - factor * iqr
    upper_bound = q3 + factor * iqr
    
    return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]

def remove_outliers_zscore(data, column, threshold=3):
    """
    Remove outliers using Z-score method.
    
    Args:
        data: pandas DataFrame
        column: column name to process
        threshold: Z-score threshold (default 3)
    
    Returns:
        DataFrame with outliers removed
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    z_scores = np.abs(stats.zscore(data[column].dropna()))
    mask = z_scores < threshold
    return data[mask]

def normalize_minmax(data, column):
    """
    Normalize data using Min-Max scaling.
    
    Args:
        data: pandas DataFrame
        column: column name to normalize
    
    Returns:
        Series with normalized values
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    col_data = data[column]
    min_val = col_data.min()
    max_val = col_data.max()
    
    if max_val == min_val:
        return pd.Series([0.5] * len(col_data), index=col_data.index)
    
    return (col_data - min_val) / (max_val - min_val)

def normalize_zscore(data, column):
    """
    Normalize data using Z-score standardization.
    
    Args:
        data: pandas DataFrame
        column: column name to normalize
    
    Returns:
        Series with standardized values
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    col_data = data[column]
    mean_val = col_data.mean()
    std_val = col_data.std()
    
    if std_val == 0:
        return pd.Series([0] * len(col_data), index=col_data.index)
    
    return (col_data - mean_val) / std_val

def clean_dataset(data, numeric_columns=None, outlier_method='iqr', normalize_method='minmax'):
    """
    Comprehensive data cleaning pipeline.
    
    Args:
        data: pandas DataFrame
        numeric_columns: list of numeric columns to process (default: all numeric)
        outlier_method: 'iqr', 'zscore', or None
        normalize_method: 'minmax', 'zscore', or None
    
    Returns:
        Cleaned DataFrame
    """
    cleaned_data = data.copy()
    
    if numeric_columns is None:
        numeric_columns = cleaned_data.select_dtypes(include=[np.number]).columns.tolist()
    
    for column in numeric_columns:
        if column not in cleaned_data.columns:
            continue
            
        # Remove outliers
        if outlier_method == 'iqr':
            cleaned_data = remove_outliers_iqr(cleaned_data, column)
        elif outlier_method == 'zscore':
            cleaned_data = remove_outliers_zscore(cleaned_data, column)
        
        # Normalize data
        if normalize_method == 'minmax':
            cleaned_data[column] = normalize_minmax(cleaned_data, column)
        elif normalize_method == 'zscore':
            cleaned_data[column] = normalize_zscore(cleaned_data, column)
    
    return cleaned_data

def validate_data(data, required_columns=None, allow_nan=False):
    """
    Validate dataset structure and content.
    
    Args:
        data: pandas DataFrame
        required_columns: list of required column names
        allow_nan: whether to allow NaN values
    
    Returns:
        tuple: (is_valid, error_message)
    """
    if not isinstance(data, pd.DataFrame):
        return False, "Input must be a pandas DataFrame"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in data.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    if not allow_nan and data.isnull().any().any():
        nan_cols = data.columns[data.isnull().any()].tolist()
        return False, f"NaN values found in columns: {nan_cols}"
    
    return True, "Data validation passed"
import pandas as pd
import numpy as np

def clean_dataset(df, drop_duplicates=True, handle_nulls='drop', null_threshold=0.5):
    """
    Clean a pandas DataFrame by handling duplicates and null values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    drop_duplicates (bool): Whether to drop duplicate rows
    handle_nulls (str): Strategy for handling nulls - 'drop', 'fill_mean', 'fill_median', 'fill_mode'
    null_threshold (float): Threshold for dropping columns with too many nulls (0.0 to 1.0)
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    
    cleaned_df = df.copy()
    
    # Drop columns with excessive null values
    null_ratio = cleaned_df.isnull().sum() / len(cleaned_df)
    cols_to_drop = null_ratio[null_ratio > null_threshold].index
    cleaned_df = cleaned_df.drop(columns=cols_to_drop)
    
    # Handle duplicates
    if drop_duplicates:
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        removed_duplicates = initial_rows - len(cleaned_df)
        print(f"Removed {removed_duplicates} duplicate rows")
    
    # Handle remaining null values
    if handle_nulls == 'drop':
        cleaned_df = cleaned_df.dropna()
    elif handle_nulls in ['fill_mean', 'fill_median', 'fill_mode']:
        for column in cleaned_df.select_dtypes(include=[np.number]).columns:
            if cleaned_df[column].isnull().any():
                if handle_nulls == 'fill_mean':
                    fill_value = cleaned_df[column].mean()
                elif handle_nulls == 'fill_median':
                    fill_value = cleaned_df[column].median()
                else:  # fill_mode
                    fill_value = cleaned_df[column].mode()[0]
                cleaned_df[column] = cleaned_df[column].fillna(fill_value)
        
        # For categorical columns, fill with mode
        for column in cleaned_df.select_dtypes(exclude=[np.number]).columns:
            if cleaned_df[column].isnull().any():
                fill_value = cleaned_df[column].mode()[0] if not cleaned_df[column].mode().empty else 'Unknown'
                cleaned_df[column] = cleaned_df[column].fillna(fill_value)
    
    # Reset index after cleaning
    cleaned_df = cleaned_df.reset_index(drop=True)
    
    print(f"Original shape: {df.shape}")
    print(f"Cleaned shape: {cleaned_df.shape}")
    print(f"Removed {len(cols_to_drop)} columns with >{null_threshold*100}% null values")
    
    return cleaned_df

def validate_dataframe(df, required_columns=None, min_rows=1):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate
    required_columns (list): List of required column names
    min_rows (int): Minimum number of rows required
    
    Returns:
    bool: True if validation passes, False otherwise
    """
    
    if not isinstance(df, pd.DataFrame):
        print("Error: Input is not a pandas DataFrame")
        return False
    
    if len(df) < min_rows:
        print(f"Error: DataFrame has less than {min_rows} rows")
        return False
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Error: Missing required columns: {missing_columns}")
            return False
    
    return True

# Example usage
if __name__ == "__main__":
    # Create sample data
    sample_data = {
        'id': [1, 2, 2, 3, 4, 5],
        'value': [10.5, 20.3, 20.3, None, 40.1, 50.0],
        'category': ['A', 'B', 'B', 'C', None, 'A'],
        'empty_column': [None, None, None, None, None, None]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    # Clean the data
    cleaned = clean_dataset(df, handle_nulls='fill_mode', null_threshold=0.3)
    print("\nCleaned DataFrame:")
    print(cleaned)
    
    # Validate the cleaned data
    is_valid = validate_dataframe(cleaned, required_columns=['id', 'value'], min_rows=1)
    print(f"\nData validation: {'PASS' if is_valid else 'FAIL'}")