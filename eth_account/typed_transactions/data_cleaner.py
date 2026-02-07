
import numpy as np
import pandas as pd

def remove_outliers_iqr(data, column, multiplier=1.5):
    """
    Remove outliers from a column using the Interquartile Range method.
    
    Parameters:
    data (pd.DataFrame): Input dataframe
    column (str): Column name to process
    multiplier (float): IQR multiplier for outlier detection
    
    Returns:
    pd.DataFrame: Dataframe with outliers removed
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in dataframe")
    
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    
    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data.copy()

def normalize_minmax(data, column):
    """
    Normalize a column using Min-Max scaling to range [0, 1].
    
    Parameters:
    data (pd.DataFrame): Input dataframe
    column (str): Column name to normalize
    
    Returns:
    pd.DataFrame: Dataframe with normalized column
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in dataframe")
    
    result = data.copy()
    min_val = result[column].min()
    max_val = result[column].max()
    
    if max_val == min_val:
        result[column] = 0.5
    else:
        result[column] = (result[column] - min_val) / (max_val - min_val)
    
    return result

def standardize_zscore(data, column):
    """
    Standardize a column using Z-score normalization.
    
    Parameters:
    data (pd.DataFrame): Input dataframe
    column (str): Column name to standardize
    
    Returns:
    pd.DataFrame: Dataframe with standardized column
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in dataframe")
    
    result = data.copy()
    mean_val = result[column].mean()
    std_val = result[column].std()
    
    if std_val == 0:
        result[column] = 0
    else:
        result[column] = (result[column] - mean_val) / std_val
    
    return result

def clean_dataset(data, numeric_columns=None, outlier_multiplier=1.5):
    """
    Comprehensive data cleaning pipeline.
    
    Parameters:
    data (pd.DataFrame): Input dataframe
    numeric_columns (list): List of numeric column names to process
    outlier_multiplier (float): IQR multiplier for outlier detection
    
    Returns:
    pd.DataFrame: Cleaned dataframe
    """
    if numeric_columns is None:
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_data = data.copy()
    
    for column in numeric_columns:
        if column in cleaned_data.columns:
            cleaned_data = remove_outliers_iqr(cleaned_data, column, outlier_multiplier)
            cleaned_data = standardize_zscore(cleaned_data, column)
    
    return cleaned_data.reset_index(drop=True)

def validate_dataframe(data, required_columns=None):
    """
    Validate dataframe structure and content.
    
    Parameters:
    data (pd.DataFrame): Dataframe to validate
    required_columns (list): List of required column names
    
    Returns:
    tuple: (is_valid, error_message)
    """
    if not isinstance(data, pd.DataFrame):
        return False, "Input must be a pandas DataFrame"
    
    if data.empty:
        return False, "DataFrame is empty"
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
    
    return True, "DataFrame is valid"import numpy as np
import pandas as pd

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def normalize_minmax(df, column):
    min_val = df[column].min()
    max_val = df[column].max()
    if max_val == min_val:
        return df[column].apply(lambda x: 0.0)
    return df[column].apply(lambda x: (x - min_val) / (max_val - min_val))

def clean_dataset(df, numeric_columns):
    cleaned_df = df.copy()
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
            cleaned_df[col] = normalize_minmax(cleaned_df, col)
    return cleaned_df.reset_index(drop=True)

def validate_data(df, required_columns):
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    return Trueimport pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range (IQR) method.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    column (str): The column name to clean.
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed from the specified column.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
    return filtered_df.reset_index(drop=True)

def clean_dataset(df, numeric_columns=None):
    """
    Clean a dataset by removing outliers from all numeric columns.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    numeric_columns (list): List of numeric column names. If None, uses all numeric columns.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame with outliers removed.
    """
    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_df = df.copy()
    
    for col in numeric_columns:
        if col in cleaned_df.columns:
            original_len = len(cleaned_df)
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
            removed_count = original_len - len(cleaned_df)
            print(f"Removed {removed_count} outliers from column '{col}'")
    
    return cleaned_df

if __name__ == "__main__":
    sample_data = {
        'id': range(1, 21),
        'value': [10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 100, 12, 13, 14, 15, 16, 17, 18, 19, -50]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print(f"\nOriginal shape: {df.shape}")
    
    cleaned_df = clean_dataset(df, ['value'])
    print(f"\nCleaned shape: {cleaned_df.shape}")
    print("\nCleaned DataFrame:")
    print(cleaned_df)
def remove_duplicates(input_list):
    """
    Remove duplicate elements from a list while preserving order.
    Returns a new list with unique elements.
    """
    seen = set()
    result = []
    for item in input_list:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result

def clean_data_with_threshold(data, threshold=None):
    """
    Clean data by removing duplicates and optionally filtering by threshold.
    If threshold is provided, only items meeting the threshold are kept.
    """
    unique_data = remove_duplicates(data)
    
    if threshold is not None:
        filtered_data = [item for item in unique_data if item >= threshold]
        return filtered_data
    
    return unique_data

if __name__ == "__main__":
    sample_data = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]
    print("Original data:", sample_data)
    print("Cleaned data:", remove_duplicates(sample_data))
    print("Cleaned with threshold 4:", clean_data_with_threshold(sample_data, 4))