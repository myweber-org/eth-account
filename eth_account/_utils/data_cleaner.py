
import numpy as np

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a dataset using the Interquartile Range method.
    
    Parameters:
    data (numpy.ndarray): Input data array
    column (int): Column index to check for outliers
    
    Returns:
    numpy.ndarray: Data with outliers removed
    """
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a numpy array")
    
    if column >= data.shape[1]:
        raise IndexError("Column index out of bounds")
    
    q1 = np.percentile(data[:, column], 25)
    q3 = np.percentile(data[:, column], 75)
    iqr = q3 - q1
    
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    mask = (data[:, column] >= lower_bound) & (data[:, column] <= upper_bound)
    return data[mask]

def calculate_statistics(data):
    """
    Calculate basic statistics for the cleaned data.
    
    Parameters:
    data (numpy.ndarray): Input data array
    
    Returns:
    dict: Dictionary containing mean, median, and std
    """
    if data.size == 0:
        return {"mean": np.nan, "median": np.nan, "std": np.nan}
    
    return {
        "mean": np.mean(data, axis=0),
        "median": np.median(data, axis=0),
        "std": np.std(data, axis=0)
    }

def validate_data(data, expected_columns):
    """
    Validate data shape and check for NaN values.
    
    Parameters:
    data (numpy.ndarray): Input data array
    expected_columns (int): Expected number of columns
    
    Returns:
    bool: True if data is valid, False otherwise
    """
    if data.shape[1] != expected_columns:
        return False
    
    if np.any(np.isnan(data)):
        return False
    
    return True

def process_dataset(data, target_column):
    """
    Main function to process dataset by removing outliers and calculating statistics.
    
    Parameters:
    data (numpy.ndarray): Input data array
    target_column (int): Column index for outlier detection
    
    Returns:
    tuple: (cleaned_data, statistics, is_valid)
    """
    if not validate_data(data, data.shape[1]):
        raise ValueError("Invalid data format")
    
    cleaned_data = remove_outliers_iqr(data, target_column)
    stats = calculate_statistics(cleaned_data)
    
    return cleaned_data, stats, Trueimport pandas as pd
import numpy as np

def remove_duplicates(df, subset=None):
    """
    Remove duplicate rows from DataFrame.
    If subset is provided, only consider certain columns for identifying duplicates.
    """
    return df.drop_duplicates(subset=subset, keep='first')

def fill_missing_values(df, strategy='mean', columns=None):
    """
    Fill missing values in DataFrame.
    Strategies: 'mean', 'median', 'mode', or 'constant' (fills with 0).
    If columns is None, applies to all numeric columns.
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    df_filled = df.copy()
    
    for col in columns:
        if col in df.columns:
            if strategy == 'mean':
                fill_value = df[col].mean()
            elif strategy == 'median':
                fill_value = df[col].median()
            elif strategy == 'mode':
                fill_value = df[col].mode()[0] if not df[col].mode().empty else 0
            elif strategy == 'constant':
                fill_value = 0
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
            
            df_filled[col] = df[col].fillna(fill_value)
    
    return df_filled

def normalize_column(df, column, method='minmax'):
    """
    Normalize a column using specified method.
    Methods: 'minmax' (0-1 scaling) or 'zscore' (standardization).
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    df_normalized = df.copy()
    
    if method == 'minmax':
        min_val = df[column].min()
        max_val = df[column].max()
        if max_val != min_val:
            df_normalized[column] = (df[column] - min_val) / (max_val - min_val)
        else:
            df_normalized[column] = 0
    
    elif method == 'zscore':
        mean_val = df[column].mean()
        std_val = df[column].std()
        if std_val != 0:
            df_normalized[column] = (df[column] - mean_val) / std_val
        else:
            df_normalized[column] = 0
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return df_normalized

def clean_dataframe(df, remove_dups=True, fill_na=True, normalize_cols=None):
    """
    Comprehensive data cleaning pipeline.
    """
    cleaned_df = df.copy()
    
    if remove_dups:
        cleaned_df = remove_duplicates(cleaned_df)
    
    if fill_na:
        cleaned_df = fill_missing_values(cleaned_df)
    
    if normalize_cols:
        for col in normalize_cols:
            if col in cleaned_df.columns:
                cleaned_df = normalize_column(cleaned_df, col)
    
    return cleaned_df

def save_cleaned_data(df, output_path, format='csv'):
    """
    Save cleaned DataFrame to file.
    Formats: 'csv', 'excel', 'json'
    """
    if format == 'csv':
        df.to_csv(output_path, index=False)
    elif format == 'excel':
        df.to_excel(output_path, index=False)
    elif format == 'json':
        df.to_json(output_path, orient='records')
    else:
        raise ValueError(f"Unsupported format: {format}")