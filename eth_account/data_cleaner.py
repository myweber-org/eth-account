import pandas as pd

def clean_dataset(df, drop_duplicates=True, fill_missing=False, fill_value=0):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean
    drop_duplicates (bool): Whether to remove duplicate rows
    fill_missing (bool): Whether to fill missing values
    fill_value: Value to use for filling missing data
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        removed = initial_rows - len(cleaned_df)
        print(f"Removed {removed} duplicate rows")
    
    if fill_missing:
        missing_before = cleaned_df.isnull().sum().sum()
        cleaned_df = cleaned_df.fillna(fill_value)
        missing_after = cleaned_df.isnull().sum().sum()
        print(f"Filled {missing_before - missing_after} missing values")
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate that a DataFrame meets basic requirements.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate
    required_columns (list): List of column names that must be present
    
    Returns:
    bool: True if validation passes, False otherwise
    """
    if not isinstance(df, pd.DataFrame):
        print("Error: Input is not a pandas DataFrame")
        return False
    
    if df.empty:
        print("Warning: DataFrame is empty")
        return True
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Error: Missing required columns: {missing_columns}")
            return False
    
    return True

def remove_outliers_iqr(df, column, multiplier=1.5):
    """
    Remove outliers from a DataFrame column using the IQR method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to process
    multiplier (float): IQR multiplier for outlier detection
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed
    """
    if column not in df.columns:
        print(f"Error: Column '{column}' not found in DataFrame")
        return df
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    initial_count = len(df)
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    removed_count = initial_count - len(filtered_df)
    
    print(f"Removed {removed_count} outliers from column '{column}'")
    return filtered_df
import pandas as pd
import numpy as np

def clean_dataframe(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    drop_duplicates (bool): Whether to drop duplicate rows. Default True.
    fill_missing (str): Method to fill missing values. 
                       Options: 'mean', 'median', 'mode', or 'drop'. Default 'mean'.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        removed = initial_rows - len(cleaned_df)
        print(f"Removed {removed} duplicate rows.")
    
    if fill_missing == 'drop':
        cleaned_df = cleaned_df.dropna()
        print("Dropped rows with missing values.")
    elif fill_missing in ['mean', 'median', 'mode']:
        numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if cleaned_df[col].isnull().any():
                if fill_missing == 'mean':
                    fill_value = cleaned_df[col].mean()
                elif fill_missing == 'median':
                    fill_value = cleaned_df[col].median()
                elif fill_missing == 'mode':
                    fill_value = cleaned_df[col].mode()[0]
                
                cleaned_df[col] = cleaned_df[col].fillna(fill_value)
                print(f"Filled missing values in column '{col}' with {fill_missing}: {fill_value:.2f}")
    
    categorical_cols = cleaned_df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if cleaned_df[col].isnull().any():
            cleaned_df[col] = cleaned_df[col].fillna('Unknown')
            print(f"Filled missing values in categorical column '{col}' with 'Unknown'")
    
    print(f"Data cleaning complete. Final shape: {cleaned_df.shape}")
    return cleaned_df

def validate_dataframe(df, required_columns=None, min_rows=1):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of required column names.
    min_rows (int): Minimum number of rows required.
    
    Returns:
    bool: True if validation passes, False otherwise.
    """
    if not isinstance(df, pd.DataFrame):
        print("Error: Input is not a pandas DataFrame.")
        return False
    
    if len(df) < min_rows:
        print(f"Error: DataFrame has fewer than {min_rows} rows.")
        return False
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            print(f"Error: Missing required columns: {missing_cols}")
            return False
    
    return True

def get_data_summary(df):
    """
    Generate a summary of the DataFrame.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame.
    
    Returns:
    dict: Summary statistics and information.
    """
    summary = {
        'shape': df.shape,
        'columns': list(df.columns),
        'dtypes': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'numeric_stats': df.describe().to_dict() if not df.select_dtypes(include=[np.number]).empty else {},
        'categorical_stats': {col: df[col].value_counts().to_dict() 
                            for col in df.select_dtypes(include=['object']).columns}
    }
    return summary

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, 2, 3, np.nan, 5],
        'B': [10, 20, 20, np.nan, 50, 60],
        'C': ['x', 'y', 'y', 'z', np.nan, 'x'],
        'D': [100, 200, 300, 400, 500, 600]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    cleaned = clean_dataframe(df, drop_duplicates=True, fill_missing='median')
    print("\nCleaned DataFrame:")
    print(cleaned)
    
    print("\nData Summary:")
    summary = get_data_summary(cleaned)
    print(f"Shape: {summary['shape']}")
    print(f"Missing values: {summary['missing_values']}")import numpy as np
import pandas as pd
from scipy import stats

def detect_outliers_iqr(data, threshold=1.5):
    """
    Detect outliers using Interquartile Range method.
    
    Args:
        data: Array-like data
        threshold: IQR multiplier for outlier detection
    
    Returns:
        Boolean mask where True indicates outliers
    """
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr
    return (data < lower_bound) | (data > upper_bound)

def remove_outliers(df, columns, threshold=1.5):
    """
    Remove outliers from specified columns in DataFrame.
    
    Args:
        df: pandas DataFrame
        columns: List of column names to process
        threshold: IQR multiplier for outlier detection
    
    Returns:
        DataFrame with outliers removed
    """
    df_clean = df.copy()
    outlier_mask = pd.Series([False] * len(df))
    
    for col in columns:
        if col in df.columns:
            col_mask = detect_outliers_iqr(df[col].values, threshold)
            outlier_mask = outlier_mask | col_mask
    
    return df_clean[~outlier_mask].reset_index(drop=True)

def normalize_minmax(data):
    """
    Normalize data using min-max scaling to [0, 1] range.
    
    Args:
        data: Array-like data
    
    Returns:
        Normalized data
    """
    data_array = np.array(data)
    min_val = np.min(data_array)
    max_val = np.max(data_array)
    
    if max_val == min_val:
        return np.zeros_like(data_array)
    
    return (data_array - min_val) / (max_val - min_val)

def standardize_zscore(data):
    """
    Standardize data using z-score normalization.
    
    Args:
        data: Array-like data
    
    Returns:
        Standardized data with mean=0, std=1
    """
    data_array = np.array(data)
    mean_val = np.mean(data_array)
    std_val = np.std(data_array)
    
    if std_val == 0:
        return np.zeros_like(data_array)
    
    return (data_array - mean_val) / std_val

def clean_dataset(df, numeric_columns, outlier_threshold=1.5, normalize=True):
    """
    Complete data cleaning pipeline.
    
    Args:
        df: Input DataFrame
        numeric_columns: List of numeric column names to process
        outlier_threshold: IQR multiplier for outlier detection
        normalize: Whether to apply min-max normalization
    
    Returns:
        Cleaned DataFrame
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame")
    
    df_clean = df.copy()
    
    if numeric_columns:
        df_clean = remove_outliers(df_clean, numeric_columns, outlier_threshold)
        
        for col in numeric_columns:
            if col in df_clean.columns:
                if normalize:
                    df_clean[col] = normalize_minmax(df_clean[col])
                else:
                    df_clean[col] = standardize_zscore(df_clean[col])
    
    return df_clean

def get_summary_statistics(df):
    """
    Calculate summary statistics for numeric columns.
    
    Args:
        df: pandas DataFrame
    
    Returns:
        Dictionary of summary statistics
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) == 0:
        return {}
    
    summary = {}
    for col in numeric_cols:
        summary[col] = {
            'mean': float(df[col].mean()),
            'median': float(df[col].median()),
            'std': float(df[col].std()),
            'min': float(df[col].min()),
            'max': float(df[col].max()),
            'q1': float(df[col].quantile(0.25)),
            'q3': float(df[col].quantile(0.75)),
            'skewness': float(stats.skew(df[col].dropna())),
            'kurtosis': float(stats.kurtosis(df[col].dropna()))
        }
    
    return summary

if __name__ == "__main__":
    sample_data = {
        'feature1': np.random.normal(50, 15, 1000),
        'feature2': np.random.exponential(10, 1000),
        'feature3': np.random.uniform(0, 100, 1000),
        'category': np.random.choice(['A', 'B', 'C'], 1000)
    }
    
    df_sample = pd.DataFrame(sample_data)
    print("Original dataset shape:", df_sample.shape)
    
    numeric_cols = ['feature1', 'feature2', 'feature3']
    df_cleaned = clean_dataset(df_sample, numeric_cols)
    print("Cleaned dataset shape:", df_cleaned.shape)
    
    stats_summary = get_summary_statistics(df_cleaned)
    print("\nSummary statistics for cleaned data:")
    for col, stats_dict in stats_summary.items():
        print(f"\n{col}:")
        for stat_name, value in stats_dict.items():
            print(f"  {stat_name}: {value:.4f}")import pandas as pd

def clean_dataset(df, drop_duplicates=True, fill_missing=None):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        drop_duplicates (bool): Whether to drop duplicate rows. Default is True.
        fill_missing (str or dict): Method to fill missing values. Can be 'mean', 'median', 
                                   'mode', or a dictionary of column:value pairs. Default is None.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    if fill_missing is not None:
        if isinstance(fill_missing, dict):
            for column, value in fill_missing.items():
                if column in cleaned_df.columns:
                    cleaned_df[column].fillna(value, inplace=True)
        elif fill_missing == 'mean':
            cleaned_df = cleaned_df.fillna(cleaned_df.mean(numeric_only=True))
        elif fill_missing == 'median':
            cleaned_df = cleaned_df.fillna(cleaned_df.median(numeric_only=True))
        elif fill_missing == 'mode':
            cleaned_df = cleaned_df.fillna(cleaned_df.mode().iloc[0])
    
    return cleaned_df

def validate_data(df, required_columns=None, min_rows=1):
    """
    Validate DataFrame structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate.
        required_columns (list): List of column names that must be present.
        min_rows (int): Minimum number of rows required.
    
    Returns:
        tuple: (is_valid, error_message)
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
import pandas as pd
import numpy as np

def clean_dataset(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    drop_duplicates (bool): Whether to remove duplicate rows.
    fill_missing (str): Method to fill missing values ('mean', 'median', 'mode', or 'drop').
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    if fill_missing == 'drop':
        cleaned_df = cleaned_df.dropna()
    elif fill_missing in ['mean', 'median']:
        numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if fill_missing == 'mean':
                cleaned_df[col].fillna(cleaned_df[col].mean(), inplace=True)
            elif fill_missing == 'median':
                cleaned_df[col].fillna(cleaned_df[col].median(), inplace=True)
    elif fill_missing == 'mode':
        for col in cleaned_df.columns:
            cleaned_df[col].fillna(cleaned_df[col].mode()[0] if not cleaned_df[col].mode().empty else None, inplace=True)
    
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
    if not isinstance(df, pd.DataFrame):
        return False, "Input is not a pandas DataFrame"
    
    if df.empty:
        return False, "DataFrame is empty"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    return True, "DataFrame is valid"

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, 2, 4, None],
        'B': [5, None, 7, 8, 9],
        'C': ['x', 'y', 'y', 'z', 'x']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nCleaned DataFrame (fill with mean):")
    cleaned = clean_dataset(df, fill_missing='mean')
    print(cleaned)
    
    is_valid, message = validate_dataframe(cleaned, required_columns=['A', 'B', 'C'])
    print(f"\nValidation: {is_valid}, Message: {message}")