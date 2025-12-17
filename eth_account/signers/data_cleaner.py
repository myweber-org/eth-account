
import numpy as np

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a specified column in a dataset using the IQR method.
    
    Parameters:
    data (numpy.ndarray): The dataset.
    column (int): Index of the column to process.
    
    Returns:
    numpy.ndarray: Dataset with outliers removed from the specified column.
    """
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a numpy array")
    
    if column >= data.shape[1] or column < 0:
        raise IndexError("Column index out of bounds")
    
    col_data = data[:, column]
    
    Q1 = np.percentile(col_data, 25)
    Q3 = np.percentile(col_data, 75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    mask = (col_data >= lower_bound) & (col_data <= upper_bound)
    
    return data[mask]
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

def normalize_column_zscore(dataframe, column):
    """
    Normalize a column using Z-score normalization.
    
    Args:
        dataframe: pandas DataFrame
        column: column name to normalize
    
    Returns:
        DataFrame with normalized column added as '{column}_normalized'
    """
    if column not in dataframe.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    normalized_col = f"{column}_normalized"
    dataframe[normalized_col] = stats.zscore(dataframe[column])
    
    return dataframe

def min_max_normalize(dataframe, column, feature_range=(0, 1)):
    """
    Normalize a column using Min-Max scaling.
    
    Args:
        dataframe: pandas DataFrame
        column: column name to normalize
        feature_range: tuple of (min, max) for output range
    
    Returns:
        DataFrame with normalized column added as '{column}_scaled'
    """
    if column not in dataframe.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    min_val = dataframe[column].min()
    max_val = dataframe[column].max()
    
    scaled_col = f"{column}_scaled"
    dataframe[scaled_col] = ((dataframe[column] - min_val) / 
                            (max_val - min_val)) * (feature_range[1] - feature_range[0]) + feature_range[0]
    
    return dataframe

def clean_dataset(dataframe, numeric_columns=None, outlier_threshold=1.5):
    """
    Comprehensive data cleaning pipeline.
    
    Args:
        dataframe: pandas DataFrame
        numeric_columns: list of numeric columns to process (default: all numeric columns)
        outlier_threshold: IQR threshold for outlier removal
    
    Returns:
        Cleaned DataFrame
    """
    if numeric_columns is None:
        numeric_columns = dataframe.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_df = dataframe.copy()
    
    for column in numeric_columns:
        if column in cleaned_df.columns:
            original_count = len(cleaned_df)
            cleaned_df = remove_outliers_iqr(cleaned_df, column, outlier_threshold)
            removed_count = original_count - len(cleaned_df)
            
            if removed_count > 0:
                print(f"Removed {removed_count} outliers from column '{column}'")
            
            cleaned_df = normalize_column_zscore(cleaned_df, column)
            cleaned_df = min_max_normalize(cleaned_df, column)
    
    return cleaned_df

def validate_dataframe(dataframe, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Args:
        dataframe: pandas DataFrame to validate
        required_columns: list of required column names
    
    Returns:
        tuple: (is_valid, error_message)
    """
    if dataframe.empty:
        return False, "DataFrame is empty"
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in dataframe.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
    
    if dataframe.isnull().sum().sum() > 0:
        return False, "DataFrame contains null values"
    
    return True, "DataFrame validation passed"

def example_usage():
    """Example usage of the data cleaning utilities."""
    np.random.seed(42)
    
    data = {
        'feature_a': np.random.normal(100, 15, 1000),
        'feature_b': np.random.exponential(50, 1000),
        'feature_c': np.random.uniform(0, 200, 1000),
        'category': np.random.choice(['A', 'B', 'C'], 1000)
    }
    
    df = pd.DataFrame(data)
    
    print("Original DataFrame shape:", df.shape)
    
    cleaned_df = clean_dataset(df, numeric_columns=['feature_a', 'feature_b', 'feature_c'])
    
    print("Cleaned DataFrame shape:", cleaned_df.shape)
    print("\nCleaned DataFrame columns:", cleaned_df.columns.tolist())
    print("\nFirst 5 rows of cleaned data:")
    print(cleaned_df.head())
    
    is_valid, message = validate_dataframe(cleaned_df)
    print(f"\nData validation: {message}")

if __name__ == "__main__":
    example_usage()