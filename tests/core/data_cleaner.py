
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range method.
    
    Args:
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

def calculate_summary_stats(df, column):
    """
    Calculate summary statistics for a DataFrame column.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        column (str): Column name to analyze
    
    Returns:
        dict: Dictionary containing summary statistics
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    stats = {
        'mean': df[column].mean(),
        'median': df[column].median(),
        'std': df[column].std(),
        'min': df[column].min(),
        'max': df[column].max(),
        'count': df[column].count(),
        'missing': df[column].isnull().sum()
    }
    
    return stats

def clean_numeric_data(df, columns=None):
    """
    Clean numeric data by removing outliers from specified columns.
    If no columns specified, clean all numeric columns.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        columns (list, optional): List of column names to clean
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_df = df.copy()
    
    for column in columns:
        if column in cleaned_df.columns:
            original_count = len(cleaned_df)
            cleaned_df = remove_outliers_iqr(cleaned_df, column)
            removed_count = original_count - len(cleaned_df)
            print(f"Removed {removed_count} outliers from column '{column}'")
    
    return cleaned_df

if __name__ == "__main__":
    sample_data = {
        'A': np.random.normal(100, 15, 1000),
        'B': np.random.exponential(50, 1000),
        'C': np.random.uniform(0, 200, 1000)
    }
    
    df = pd.DataFrame(sample_data)
    df.loc[::100, 'A'] = 500
    
    print("Original DataFrame shape:", df.shape)
    print("\nOriginal summary for column 'A':")
    print(calculate_summary_stats(df, 'A'))
    
    cleaned_df = clean_numeric_data(df, ['A'])
    
    print("\nCleaned DataFrame shape:", cleaned_df.shape)
    print("\nCleaned summary for column 'A':")
    print(calculate_summary_stats(cleaned_df, 'A'))
import numpy as np
import pandas as pd

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return filtered_df

def clean_dataset(df, numeric_columns):
    cleaned_df = df.copy()
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
    cleaned_df = cleaned_df.dropna()
    cleaned_df = cleaned_df.reset_index(drop=True)
    return cleaned_df

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'A': np.random.normal(0, 1, 100),
        'B': np.random.exponential(2, 100),
        'C': np.random.randint(1, 50, 100)
    })
    sample_data.loc[5, 'A'] = 100
    sample_data.loc[10, 'B'] = 200
    cleaned = clean_dataset(sample_data, ['A', 'B'])
    print(f"Original shape: {sample_data.shape}")
    print(f"Cleaned shape: {cleaned.shape}")
import pandas as pd
import numpy as np
from scipy import stats

def normalize_data(df, columns=None, method='zscore'):
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    df_normalized = df.copy()
    
    for col in columns:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            if method == 'zscore':
                df_normalized[col] = stats.zscore(df[col])
            elif method == 'minmax':
                min_val = df[col].min()
                max_val = df[col].max()
                if max_val != min_val:
                    df_normalized[col] = (df[col] - min_val) / (max_val - min_val)
                else:
                    df_normalized[col] = 0
            elif method == 'robust':
                median = df[col].median()
                iqr = df[col].quantile(0.75) - df[col].quantile(0.25)
                if iqr != 0:
                    df_normalized[col] = (df[col] - median) / iqr
                else:
                    df_normalized[col] = 0
    
    return df_normalized

def remove_outliers(df, columns=None, method='iqr', threshold=1.5):
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    df_clean = df.copy()
    
    for col in columns:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            if method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                mask = (df[col] >= lower_bound) & (df[col] <= upper_bound)
                df_clean = df_clean[mask]
            elif method == 'zscore':
                z_scores = np.abs(stats.zscore(df[col]))
                mask = z_scores < threshold
                df_clean = df_clean[mask]
    
    return df_clean.reset_index(drop=True)

def handle_missing_values(df, columns=None, strategy='mean'):
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    df_filled = df.copy()
    
    for col in columns:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            if strategy == 'mean':
                fill_value = df[col].mean()
            elif strategy == 'median':
                fill_value = df[col].median()
            elif strategy == 'mode':
                fill_value = df[col].mode()[0] if not df[col].mode().empty else 0
            elif strategy == 'constant':
                fill_value = 0
            else:
                fill_value = df[col].mean()
            
            df_filled[col] = df[col].fillna(fill_value)
    
    return df_filled

def clean_dataset(df, normalize=True, remove_outliers_flag=True, handle_missing=True):
    df_processed = df.copy()
    
    if handle_missing:
        df_processed = handle_missing_values(df_processed)
    
    if remove_outliers_flag:
        df_processed = remove_outliers(df_processed)
    
    if normalize:
        df_processed = normalize_data(df_processed)
    
    return df_processed
import pandas as pd
import numpy as np

def clean_dataset(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    drop_duplicates (bool): Whether to drop duplicate rows. Default is True.
    fill_missing (str): Method to fill missing values. Options: 'mean', 'median', 'mode', 'drop', or 'zero'. Default is 'mean'.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    if fill_missing == 'drop':
        cleaned_df = cleaned_df.dropna()
    elif fill_missing == 'zero':
        cleaned_df = cleaned_df.fillna(0)
    elif fill_missing == 'mean':
        for column in cleaned_df.select_dtypes(include=[np.number]).columns:
            cleaned_df[column] = cleaned_df[column].fillna(cleaned_df[column].mean())
    elif fill_missing == 'median':
        for column in cleaned_df.select_dtypes(include=[np.number]).columns:
            cleaned_df[column] = cleaned_df[column].fillna(cleaned_df[column].median())
    elif fill_missing == 'mode':
        for column in cleaned_df.columns:
            cleaned_df[column] = cleaned_df[column].fillna(cleaned_df[column].mode()[0] if not cleaned_df[column].mode().empty else None)
    
    return cleaned_df

def validate_data(df, required_columns=None, min_rows=1):
    """
    Validate the DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of column names that must be present.
    min_rows (int): Minimum number of rows required.
    
    Returns:
    tuple: (bool, str) indicating success and message.
    """
    if df.empty:
        return False, "DataFrame is empty"
    
    if len(df) < min_rows:
        return False, f"DataFrame has fewer than {min_rows} rows"
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
    
    return True, "Data validation passed"

def remove_outliers(df, column, method='iqr', threshold=1.5):
    """
    Remove outliers from a specific column using specified method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame.
    column (str): Column name to process.
    method (str): Method for outlier detection: 'iqr' or 'zscore'.
    threshold (float): Threshold for outlier detection.
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    if not pd.api.types.is_numeric_dtype(df[column]):
        raise ValueError(f"Column '{column}' must be numeric")
    
    data = df[column].copy()
    
    if method == 'iqr':
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        mask = (data >= lower_bound) & (data <= upper_bound)
    elif method == 'zscore':
        z_scores = np.abs((data - data.mean()) / data.std())
        mask = z_scores <= threshold
    else:
        raise ValueError("Method must be 'iqr' or 'zscore'")
    
    return df[mask]

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, 2, 4, 5, None, 7, 8, 9, 100],
        'B': [10, 20, 20, 40, 50, 60, 70, 80, 90, 1000],
        'C': ['a', 'b', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nCleaned DataFrame (drop duplicates, fill with mean):")
    cleaned = clean_dataset(df, drop_duplicates=True, fill_missing='mean')
    print(cleaned)
    
    print("\nValidating cleaned data:")
    is_valid, message = validate_data(cleaned, required_columns=['A', 'B', 'C'], min_rows=5)
    print(f"Validation: {is_valid}, Message: {message}")
    
    print("\nRemoving outliers from column 'B' using IQR method:")
    no_outliers = remove_outliers(cleaned, 'B', method='iqr', threshold=1.5)
    print(no_outliers)