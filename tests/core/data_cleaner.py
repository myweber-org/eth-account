
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column, factor=1.5):
    """
    Remove outliers using IQR method
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    
    lower_bound = q1 - factor * iqr
    upper_bound = q3 + factor * iqr
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    removed_count = len(data) - len(filtered_data)
    
    return filtered_data, removed_count

def normalize_minmax(data, column):
    """
    Normalize data using min-max scaling
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    min_val = data[column].min()
    max_val = data[column].max()
    
    if max_val == min_val:
        return data[column].apply(lambda x: 0.5)
    
    normalized = (data[column] - min_val) / (max_val - min_val)
    return normalized

def standardize_zscore(data, column):
    """
    Standardize data using z-score method
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    mean_val = data[column].mean()
    std_val = data[column].std()
    
    if std_val == 0:
        return data[column].apply(lambda x: 0)
    
    standardized = (data[column] - mean_val) / std_val
    return standardized

def clean_dataset(df, numeric_columns=None, outlier_factor=1.5):
    """
    Comprehensive data cleaning function
    """
    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_df = df.copy()
    stats_report = {}
    
    for col in numeric_columns:
        if col in df.columns:
            # Remove outliers
            cleaned_df, removed = remove_outliers_iqr(cleaned_df, col, outlier_factor)
            
            # Normalize
            cleaned_df[f"{col}_normalized"] = normalize_minmax(cleaned_df, col)
            
            # Standardize
            cleaned_df[f"{col}_standardized"] = standardize_zscore(cleaned_df, col)
            
            stats_report[col] = {
                'original_rows': len(df),
                'cleaned_rows': len(cleaned_df),
                'outliers_removed': removed,
                'normalized_range': (cleaned_df[f"{col}_normalized"].min(), 
                                    cleaned_df[f"{col}_normalized"].max()),
                'standardized_mean': cleaned_df[f"{col}_standardized"].mean(),
                'standardized_std': cleaned_df[f"{col}_standardized"].std()
            }
    
    return cleaned_df, stats_report

def validate_data(df, required_columns=None, allow_nan=False):
    """
    Validate dataset structure and content
    """
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
    
    if not allow_nan:
        nan_count = df.isnull().sum().sum()
        if nan_count > 0:
            raise ValueError(f"Dataset contains {nan_count} NaN values")
    
    return True

def example_usage():
    """
    Example usage of the data cleaning functions
    """
    # Create sample data
    np.random.seed(42)
    data = {
        'feature_a': np.random.normal(100, 15, 1000),
        'feature_b': np.random.exponential(50, 1000),
        'category': np.random.choice(['A', 'B', 'C'], 1000)
    }
    
    df = pd.DataFrame(data)
    
    # Add some outliers
    df.loc[1000] = {'feature_a': 500, 'feature_b': 1000, 'category': 'A'}
    df.loc[1001] = {'feature_a': -200, 'feature_b': 5, 'category': 'B'}
    
    # Clean the dataset
    cleaned_df, report = clean_dataset(df, ['feature_a', 'feature_b'])
    
    print(f"Original shape: {df.shape}")
    print(f"Cleaned shape: {cleaned_df.shape}")
    
    for col, stats in report.items():
        print(f"\n{col}:")
        print(f"  Outliers removed: {stats['outliers_removed']}")
        print(f"  Normalized range: {stats['normalized_range']}")
    
    return cleaned_df, report

if __name__ == "__main__":
    cleaned_data, statistics = example_usage()
import pandas as pd
import numpy as np

def clean_missing_values(df, strategy='mean', columns=None):
    """
    Handle missing values in a DataFrame.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame.
    strategy (str): Strategy for imputation ('mean', 'median', 'mode', 'drop').
    columns (list): List of columns to process, if None processes all numeric columns.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    df_clean = df.copy()
    
    if columns is None:
        columns = df_clean.select_dtypes(include=[np.number]).columns.tolist()
    
    if strategy == 'drop':
        df_clean = df_clean.dropna(subset=columns)
    else:
        for col in columns:
            if col in df_clean.columns:
                if strategy == 'mean':
                    fill_value = df_clean[col].mean()
                elif strategy == 'median':
                    fill_value = df_clean[col].median()
                elif strategy == 'mode':
                    fill_value = df_clean[col].mode()[0]
                else:
                    raise ValueError(f"Unsupported strategy: {strategy}")
                
                df_clean[col] = df_clean[col].fillna(fill_value)
    
    return df_clean

def remove_outliers_iqr(df, columns=None, threshold=1.5):
    """
    Remove outliers using the Interquartile Range (IQR) method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame.
    columns (list): List of columns to process, if None processes all numeric columns.
    threshold (float): Multiplier for IQR.
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed.
    """
    df_clean = df.copy()
    
    if columns is None:
        columns = df_clean.select_dtypes(include=[np.number]).columns.tolist()
    
    for col in columns:
        if col in df_clean.columns:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
    
    return df_clean.reset_index(drop=True)

def standardize_columns(df, columns=None):
    """
    Standardize numeric columns to have zero mean and unit variance.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame.
    columns (list): List of columns to standardize.
    
    Returns:
    pd.DataFrame: DataFrame with standardized columns.
    """
    df_standardized = df.copy()
    
    if columns is None:
        columns = df_standardized.select_dtypes(include=[np.number]).columns.tolist()
    
    for col in columns:
        if col in df_standardized.columns:
            mean = df_standardized[col].mean()
            std = df_standardized[col].std()
            if std > 0:
                df_standardized[col] = (df_standardized[col] - mean) / std
    
    return df_standardized

def get_data_summary(df):
    """
    Generate a summary of the DataFrame including missing values and basic statistics.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame.
    
    Returns:
    dict: Summary statistics.
    """
    summary = {
        'shape': df.shape,
        'missing_values': df.isnull().sum().to_dict(),
        'dtypes': df.dtypes.astype(str).to_dict(),
        'numeric_stats': df.describe().to_dict() if not df.select_dtypes(include=[np.number]).empty else {}
    }
    return summary

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, np.nan, 4, 5, 100],
        'B': [10, 20, 30, np.nan, 50, 60],
        'C': ['x', 'y', 'z', 'x', 'y', 'z']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nData Summary:")
    print(get_data_summary(df))
    
    df_clean = clean_missing_values(df, strategy='mean')
    print("\nAfter cleaning missing values:")
    print(df_clean)
    
    df_no_outliers = remove_outliers_iqr(df_clean, threshold=1.5)
    print("\nAfter removing outliers:")
    print(df_no_outliers)
    
    df_standardized = standardize_columns(df_no_outliers)
    print("\nAfter standardization:")
    print(df_standardized)
def remove_duplicates(input_list):
    """
    Remove duplicate items from a list while preserving order.
    
    Args:
        input_list (list): List containing potentially duplicate items.
    
    Returns:
        list: List with duplicates removed.
    """
    seen = set()
    result = []
    
    for item in input_list:
        if item not in seen:
            seen.add(item)
            result.append(item)
    
    return result

def clean_numeric_data(values, default=0):
    """
    Clean numeric data by converting strings to floats and handling invalid values.
    
    Args:
        values (list): List of values to clean.
        default (float): Default value for invalid entries.
    
    Returns:
        list: Cleaned list of numeric values.
    """
    cleaned = []
    
    for value in values:
        try:
            cleaned.append(float(value))
        except (ValueError, TypeError):
            cleaned.append(default)
    
    return cleaned

def filter_by_threshold(data, threshold, key=None):
    """
    Filter data based on a threshold value.
    
    Args:
        data (list): List of data points to filter.
        threshold (float): Threshold value for filtering.
        key (callable, optional): Function to extract value from data points.
    
    Returns:
        list: Filtered data points.
    """
    if key is None:
        key = lambda x: x
    
    return [item for item in data if key(item) >= threshold]

if __name__ == "__main__":
    # Example usage
    sample_data = [1, 2, 2, 3, 4, 4, 5]
    cleaned = remove_duplicates(sample_data)
    print(f"Original: {sample_data}")
    print(f"Cleaned: {cleaned}")
    
    numeric_data = ["1.5", "2.3", "invalid", "4.7", None]
    cleaned_numeric = clean_numeric_data(numeric_data)
    print(f"Numeric data: {cleaned_numeric}")
    
    values = [10, 25, 5, 30, 15]
    filtered = filter_by_threshold(values, threshold=20)
    print(f"Filtered values: {filtered}")
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
    
    return filtered_df.reset_index(drop=True)

def calculate_summary_statistics(df, column):
    """
    Calculate summary statistics for a column after outlier removal.
    
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
        'count': len(df[column])
    }
    
    return stats

def process_dataframe(df, numeric_columns):
    """
    Process multiple numeric columns to remove outliers.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        numeric_columns (list): List of column names to process
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    for column in numeric_columns:
        if column in cleaned_df.columns and pd.api.types.is_numeric_dtype(cleaned_df[column]):
            cleaned_df = remove_outliers_iqr(cleaned_df, column)
    
    return cleaned_df

if __name__ == "__main__":
    sample_data = {
        'temperature': [22, 23, 24, 25, 26, 100, 27, 28, 29, -10],
        'humidity': [45, 46, 47, 48, 49, 50, 51, 52, 53, 54],
        'pressure': [1013, 1014, 1015, 1016, 1017, 1018, 1019, 1020, 1021, 1022]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nSummary statistics for temperature:")
    print(calculate_summary_statistics(df, 'temperature'))
    
    cleaned_df = process_dataframe(df, ['temperature'])
    print("\nCleaned DataFrame:")
    print(cleaned_df)
    print("\nSummary statistics after cleaning:")
    print(calculate_summary_statistics(cleaned_df, 'temperature'))
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

def calculate_basic_stats(df, column):
    """
    Calculate basic statistics for a column.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        column (str): Column name
    
    Returns:
        dict: Dictionary containing statistics
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    stats = {
        'mean': df[column].mean(),
        'median': df[column].median(),
        'std': df[column].std(),
        'min': df[column].min(),
        'max': df[column].max(),
        'count': df[column].count()
    }
    
    return stats

def clean_numeric_data(df, columns=None):
    """
    Clean numeric data by removing NaN values and converting to appropriate types.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        columns (list, optional): List of columns to clean. If None, clean all numeric columns.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_df = df.copy()
    
    for col in columns:
        if col in cleaned_df.columns:
            cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='coerce')
            cleaned_df = cleaned_df.dropna(subset=[col])
    
    return cleaned_df