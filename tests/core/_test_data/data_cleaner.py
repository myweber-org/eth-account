import numpy as np
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
        return df[column].apply(lambda x: 0.5)
    return df[column].apply(lambda x: (x - min_val) / (max_val - min_val))

def clean_dataset(df, numeric_columns):
    cleaned_df = df.copy()
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
            cleaned_df[col] = normalize_minmax(cleaned_df, col)
    return cleaned_df.reset_index(drop=True)

def validate_dataframe(df):
    required_checks = [
        lambda x: isinstance(x, pd.DataFrame),
        lambda x: not x.empty,
        lambda x: x.isnull().sum().sum() == 0
    ]
    for check in required_checks:
        if not check(df):
            return False
    return True

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'feature_a': np.random.normal(100, 15, 50),
        'feature_b': np.random.uniform(0, 1, 50),
        'category': np.random.choice(['X', 'Y', 'Z'], 50)
    })
    
    print("Original shape:", sample_data.shape)
    cleaned = clean_dataset(sample_data, ['feature_a', 'feature_b'])
    print("Cleaned shape:", cleaned.shape)
    print("Validation passed:", validate_dataframe(cleaned))
import numpy as np

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a specified column using the Interquartile Range method.
    
    Parameters:
    data (list or np.array): Input data array
    column (int): Column index to process (for 2D arrays)
    
    Returns:
    np.array: Data with outliers removed
    """
    if isinstance(data, list):
        data = np.array(data)
    
    # Handle 2D arrays
    if data.ndim == 2:
        column_data = data[:, column]
    else:
        column_data = data
    
    # Calculate IQR
    q1 = np.percentile(column_data, 25)
    q3 = np.percentile(column_data, 75)
    iqr = q3 - q1
    
    # Define outlier bounds
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    # Filter data
    if data.ndim == 2:
        mask = (column_data >= lower_bound) & (column_data <= upper_bound)
        return data[mask]
    else:
        return column_data[(column_data >= lower_bound) & (column_data <= upper_bound)]

def calculate_statistics(data):
    """
    Calculate basic statistics for the data.
    
    Parameters:
    data (np.array): Input data array
    
    Returns:
    dict: Dictionary containing statistics
    """
    if isinstance(data, list):
        data = np.array(data)
    
    stats = {
        'mean': np.mean(data),
        'median': np.median(data),
        'std': np.std(data),
        'min': np.min(data),
        'max': np.max(data),
        'count': len(data)
    }
    
    return stats

def normalize_data(data, method='minmax'):
    """
    Normalize data using specified method.
    
    Parameters:
    data (np.array): Input data array
    method (str): Normalization method ('minmax' or 'zscore')
    
    Returns:
    np.array: Normalized data
    """
    if isinstance(data, list):
        data = np.array(data)
    
    if method == 'minmax':
        data_min = np.min(data)
        data_max = np.max(data)
        if data_max - data_min == 0:
            return np.zeros_like(data)
        return (data - data_min) / (data_max - data_min)
    
    elif method == 'zscore':
        data_mean = np.mean(data)
        data_std = np.std(data)
        if data_std == 0:
            return np.zeros_like(data)
        return (data - data_mean) / data_std
    
    else:
        raise ValueError("Method must be 'minmax' or 'zscore'")

def example_usage():
    """Example usage of the data cleaning functions."""
    # Generate sample data with outliers
    np.random.seed(42)
    clean_data = np.random.normal(100, 15, 95)
    outlier_data = np.random.normal(200, 30, 5)
    sample_data = np.concatenate([clean_data, outlier_data])
    
    print("Original data statistics:")
    original_stats = calculate_statistics(sample_data)
    for key, value in original_stats.items():
        print(f"{key}: {value:.2f}")
    
    # Remove outliers
    cleaned_data = remove_outliers_iqr(sample_data, None)
    
    print("\nCleaned data statistics:")
    cleaned_stats = calculate_statistics(cleaned_data)
    for key, value in cleaned_stats.items():
        print(f"{key}: {value:.2f}")
    
    # Normalize data
    normalized_data = normalize_data(cleaned_data, method='zscore')
    
    print(f"\nNormalized data (first 5 values): {normalized_data[:5]}")
    
    return cleaned_data, normalized_data

if __name__ == "__main__":
    example_usage()
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
        'count': df[column].count()
    }
    
    return stats

def process_dataframe(df, numeric_columns):
    """
    Process multiple numeric columns to remove outliers and return cleaned DataFrame.
    
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
        'temperature': [22, 23, 24, 25, 26, 27, 28, 29, 30, 100],
        'humidity': [45, 46, 47, 48, 49, 50, 51, 52, 53, 54],
        'pressure': [1013, 1012, 1014, 1015, 1016, 1017, 1018, 1019, 1020, 1100]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print()
    
    cleaned_df = process_dataframe(df, ['temperature', 'pressure'])
    print("Cleaned DataFrame:")
    print(cleaned_df)
    print()
    
    for column in ['temperature', 'pressure']:
        stats = calculate_summary_statistics(cleaned_df, column)
        print(f"Statistics for {column}:")
        for key, value in stats.items():
            print(f"  {key}: {value:.2f}")
        print()
import numpy as np
import pandas as pd
from scipy import stats

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.original_shape = df.shape
        
    def remove_outliers_iqr(self, columns=None, factor=1.5):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        clean_df = self.df.copy()
        for col in columns:
            if col in clean_df.columns:
                Q1 = clean_df[col].quantile(0.25)
                Q3 = clean_df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - factor * IQR
                upper_bound = Q3 + factor * IQR
                clean_df = clean_df[(clean_df[col] >= lower_bound) & (clean_df[col] <= upper_bound)]
        
        self.df = clean_df
        return self
    
    def remove_outliers_zscore(self, columns=None, threshold=3):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        clean_df = self.df.copy()
        for col in columns:
            if col in clean_df.columns:
                z_scores = np.abs(stats.zscore(clean_df[col].dropna()))
                clean_df = clean_df[(z_scores < threshold) | clean_df[col].isna()]
        
        self.df = clean_df
        return self
    
    def normalize_minmax(self, columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        normalized_df = self.df.copy()
        for col in columns:
            if col in normalized_df.columns:
                col_min = normalized_df[col].min()
                col_max = normalized_df[col].max()
                if col_max != col_min:
                    normalized_df[col] = (normalized_df[col] - col_min) / (col_max - col_min)
        
        self.df = normalized_df
        return self
    
    def normalize_zscore(self, columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        normalized_df = self.df.copy()
        for col in columns:
            if col in normalized_df.columns:
                col_mean = normalized_df[col].mean()
                col_std = normalized_df[col].std()
                if col_std > 0:
                    normalized_df[col] = (normalized_df[col] - col_mean) / col_std
        
        self.df = normalized_df
        return self
    
    def fill_missing_mean(self, columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        filled_df = self.df.copy()
        for col in columns:
            if col in filled_df.columns:
                filled_df[col] = filled_df[col].fillna(filled_df[col].mean())
        
        self.df = filled_df
        return self
    
    def fill_missing_median(self, columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        filled_df = self.df.copy()
        for col in columns:
            if col in filled_df.columns:
                filled_df[col] = filled_df[col].fillna(filled_df[col].median())
        
        self.df = filled_df
        return self
    
    def get_cleaned_data(self):
        return self.df
    
    def get_removed_count(self):
        return self.original_shape[0] - self.df.shape[0]
    
    def get_summary(self):
        summary = {
            'original_rows': self.original_shape[0],
            'cleaned_rows': self.df.shape[0],
            'rows_removed': self.get_removed_count(),
            'original_columns': self.original_shape[1],
            'cleaned_columns': self.df.shape[1]
        }
        return summary

def clean_dataset(df, method='iqr', normalize=False, fill_missing=True):
    cleaner = DataCleaner(df)
    
    if method == 'iqr':
        cleaner.remove_outliers_iqr()
    elif method == 'zscore':
        cleaner.remove_outliers_zscore()
    
    if fill_missing:
        cleaner.fill_missing_mean()
    
    if normalize:
        cleaner.normalize_minmax()
    
    return cleaner.get_cleaned_data(), cleaner.get_summary()
import numpy as np

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a pandas DataFrame column using the IQR method.
    
    Parameters:
    data (pd.DataFrame): The DataFrame containing the data.
    column (str): The column name to process.
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed from the specified column.
    """
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data

def calculate_summary_statistics(data, column):
    """
    Calculate summary statistics for a column after outlier removal.
    
    Parameters:
    data (pd.DataFrame): The DataFrame containing the data.
    column (str): The column name to analyze.
    
    Returns:
    dict: Dictionary containing count, mean, std, min, and max.
    """
    stats = {
        'count': data[column].count(),
        'mean': data[column].mean(),
        'std': data[column].std(),
        'min': data[column].min(),
        'max': data[column].max()
    }
    return stats

if __name__ == "__main__":
    import pandas as pd
    
    sample_data = pd.DataFrame({
        'values': [10, 12, 12, 13, 12, 11, 10, 100, 12, 14, 13, 12, 11, 10, 9, 8, 12, 13, 14, 15]
    })
    
    print("Original data:")
    print(sample_data)
    
    cleaned_data = remove_outliers_iqr(sample_data, 'values')
    print("\nData after removing outliers:")
    print(cleaned_data)
    
    stats = calculate_summary_statistics(cleaned_data, 'values')
    print("\nSummary statistics:")
    for key, value in stats.items():
        print(f"{key}: {value:.2f}")
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column, multiplier=1.5):
    """
    Remove outliers using the Interquartile Range method.
    
    Args:
        data: pandas DataFrame
        column: column name to process
        multiplier: IQR multiplier for outlier detection
    
    Returns:
        DataFrame with outliers removed
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data

def remove_outliers_zscore(data, column, threshold=3):
    """
    Remove outliers using Z-score method.
    
    Args:
        data: pandas DataFrame
        column: column name to process
        threshold: Z-score threshold for outlier detection
    
    Returns:
        DataFrame with outliers removed
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    z_scores = np.abs(stats.zscore(data[column]))
    filtered_data = data[z_scores < threshold]
    return filtered_data

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
    
    min_val = data[column].min()
    max_val = data[column].max()
    
    if max_val == min_val:
        return pd.Series([0.5] * len(data), index=data.index)
    
    normalized = (data[column] - min_val) / (max_val - min_val)
    return normalized

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
    
    mean_val = data[column].mean()
    std_val = data[column].std()
    
    if std_val == 0:
        return pd.Series([0] * len(data), index=data.index)
    
    standardized = (data[column] - mean_val) / std_val
    return standardized

def clean_dataset(data, numeric_columns=None, outlier_method='iqr', normalize_method='minmax'):
    """
    Comprehensive data cleaning pipeline.
    
    Args:
        data: pandas DataFrame
        numeric_columns: list of numeric columns to process (defaults to all numeric columns)
        outlier_method: 'iqr' or 'zscore' for outlier removal
        normalize_method: 'minmax' or 'zscore' for normalization
    
    Returns:
        Cleaned DataFrame
    """
    if numeric_columns is None:
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_data = data.copy()
    
    for column in numeric_columns:
        if column not in cleaned_data.columns:
            continue
            
        if outlier_method == 'iqr':
            cleaned_data = remove_outliers_iqr(cleaned_data, column)
        elif outlier_method == 'zscore':
            cleaned_data = remove_outliers_zscore(cleaned_data, column)
        
        if normalize_method == 'minmax':
            cleaned_data[f'{column}_normalized'] = normalize_minmax(cleaned_data, column)
        elif normalize_method == 'zscore':
            cleaned_data[f'{column}_standardized'] = normalize_zscore(cleaned_data, column)
    
    return cleaned_data

def get_statistics(data, column):
    """
    Calculate descriptive statistics for a column.
    
    Args:
        data: pandas DataFrame
        column: column name
    
    Returns:
        Dictionary with statistics
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    stats_dict = {
        'count': data[column].count(),
        'mean': data[column].mean(),
        'std': data[column].std(),
        'min': data[column].min(),
        '25%': data[column].quantile(0.25),
        'median': data[column].median(),
        '75%': data[column].quantile(0.75),
        'max': data[column].max(),
        'skewness': data[column].skew(),
        'kurtosis': data[column].kurtosis()
    }
    
    return stats_dictimport pandas as pd
import numpy as np

def clean_dataframe(df, missing_strategy='mean', outlier_threshold=3):
    """
    Clean a pandas DataFrame by handling missing values and outliers.

    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    missing_strategy (str): Strategy for handling missing values.
                            Options: 'mean', 'median', 'mode', 'drop'.
    outlier_threshold (float): Number of standard deviations to consider as outlier.

    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()

    # Handle missing values
    if missing_strategy == 'mean':
        cleaned_df = cleaned_df.fillna(cleaned_df.mean(numeric_only=True))
    elif missing_strategy == 'median':
        cleaned_df = cleaned_df.fillna(cleaned_df.median(numeric_only=True))
    elif missing_strategy == 'mode':
        cleaned_df = cleaned_df.fillna(cleaned_df.mode().iloc[0])
    elif missing_strategy == 'drop':
        cleaned_df = cleaned_df.dropna()
    else:
        raise ValueError("Invalid missing_strategy. Choose from 'mean', 'median', 'mode', 'drop'.")

    # Handle outliers for numeric columns
    numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        col_mean = cleaned_df[col].mean()
        col_std = cleaned_df[col].std()
        if col_std > 0:  # Avoid division by zero
            z_scores = np.abs((cleaned_df[col] - col_mean) / col_std)
            cleaned_df.loc[z_scores > outlier_threshold, col] = np.nan

    # Fill outlier-induced missing values with column mean
    cleaned_df = cleaned_df.fillna(cleaned_df.mean(numeric_only=True))

    return cleaned_df

def validate_dataframe(df):
    """
    Validate DataFrame for basic integrity checks.

    Parameters:
    df (pd.DataFrame): DataFrame to validate.

    Returns:
    dict: Dictionary containing validation results.
    """
    validation_results = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': df.isnull().sum().sum(),
        'duplicate_rows': df.duplicated().sum(),
        'numeric_columns': list(df.select_dtypes(include=[np.number]).columns),
        'non_numeric_columns': list(df.select_dtypes(exclude=[np.number]).columns)
    }
    return validation_results

# Example usage (commented out for production)
# if __name__ == "__main__":
#     sample_data = pd.DataFrame({
#         'A': [1, 2, np.nan, 4, 100],
#         'B': [5, 6, 7, np.nan, 9],
#         'C': ['x', 'y', 'z', 'x', 'y']
#     })
#     
#     print("Original DataFrame:")
#     print(sample_data)
#     
#     cleaned = clean_dataframe(sample_data, missing_strategy='mean')
#     print("\nCleaned DataFrame:")
#     print(cleaned)
#     
#     validation = validate_dataframe(cleaned)
#     print("\nValidation Results:")
#     for key, value in validation.items():
#         print(f"{key}: {value}")