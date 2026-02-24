
import pandas as pd
import numpy as np
from typing import Optional, List

class DataCleaner:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.original_shape = df.shape
        
    def remove_duplicates(self, subset: Optional[List[str]] = None) -> 'DataCleaner':
        self.df = self.df.drop_duplicates(subset=subset, keep='first')
        return self
        
    def handle_missing_values(self, strategy: str = 'drop', fill_value: Optional[float] = None) -> 'DataCleaner':
        if strategy == 'drop':
            self.df = self.df.dropna()
        elif strategy == 'fill':
            if fill_value is not None:
                self.df = self.df.fillna(fill_value)
            else:
                self.df = self.df.fillna(self.df.mean())
        return self
        
    def normalize_column(self, column: str) -> 'DataCleaner':
        if column in self.df.columns:
            col_min = self.df[column].min()
            col_max = self.df[column].max()
            if col_max > col_min:
                self.df[column] = (self.df[column] - col_min) / (col_max - col_min)
        return self
        
    def get_cleaned_data(self) -> pd.DataFrame:
        return self.df
        
    def get_summary(self) -> dict:
        cleaned_shape = self.df.shape
        return {
            'original_rows': self.original_shape[0],
            'original_columns': self.original_shape[1],
            'cleaned_rows': cleaned_shape[0],
            'cleaned_columns': cleaned_shape[1],
            'rows_removed': self.original_shape[0] - cleaned_shape[0],
            'null_count': self.df.isnull().sum().sum()
        }

def clean_dataset(df: pd.DataFrame, 
                  remove_dups: bool = True,
                  handle_nulls: str = 'drop',
                  normalize_cols: Optional[List[str]] = None) -> pd.DataFrame:
    
    cleaner = DataCleaner(df)
    
    if remove_dups:
        cleaner.remove_duplicates()
        
    cleaner.handle_missing_values(strategy=handle_nulls)
    
    if normalize_cols:
        for col in normalize_cols:
            cleaner.normalize_column(col)
    
    return cleaner.get_cleaned_data()
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to clean
    
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
    Calculate basic statistics for a DataFrame column.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to analyze
    
    Returns:
    dict: Dictionary containing statistical measures
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

def clean_dataset(df, numeric_columns=None):
    """
    Clean a dataset by removing outliers from all numeric columns.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    numeric_columns (list): List of numeric column names. If None, uses all numeric columns.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_df = df.copy()
    
    for column in numeric_columns:
        if column in df.columns:
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
    
    print("Original dataset shape:", df.shape)
    print("\nOriginal statistics for column 'A':")
    print(calculate_basic_stats(df, 'A'))
    
    cleaned_df = clean_dataset(df, ['A', 'B'])
    
    print("\nCleaned dataset shape:", cleaned_df.shape)
    print("\nCleaned statistics for column 'A':")
    print(calculate_basic_stats(cleaned_df, 'A'))
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(dataframe, column, threshold=1.5):
    """
    Remove outliers from a DataFrame column using IQR method.
    
    Parameters:
    dataframe (pd.DataFrame): Input DataFrame
    column (str): Column name to process
    threshold (float): IQR multiplier for outlier detection
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed
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
    
    return filtered_df.copy()

def z_score_normalize(dataframe, columns=None):
    """
    Apply Z-score normalization to specified columns.
    
    Parameters:
    dataframe (pd.DataFrame): Input DataFrame
    columns (list): List of column names to normalize. If None, normalize all numeric columns.
    
    Returns:
    pd.DataFrame: DataFrame with normalized columns
    """
    if columns is None:
        numeric_cols = dataframe.select_dtypes(include=[np.number]).columns
        columns = list(numeric_cols)
    
    normalized_df = dataframe.copy()
    
    for col in columns:
        if col not in normalized_df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame")
        
        if normalized_df[col].dtype in [np.float64, np.int64]:
            mean_val = normalized_df[col].mean()
            std_val = normalized_df[col].std()
            
            if std_val > 0:
                normalized_df[col] = (normalized_df[col] - mean_val) / std_val
            else:
                normalized_df[col] = 0
    
    return normalized_df

def min_max_normalize(dataframe, columns=None, feature_range=(0, 1)):
    """
    Apply Min-Max normalization to specified columns.
    
    Parameters:
    dataframe (pd.DataFrame): Input DataFrame
    columns (list): List of column names to normalize
    feature_range (tuple): Desired range of transformed data
    
    Returns:
    pd.DataFrame: DataFrame with normalized columns
    """
    if columns is None:
        numeric_cols = dataframe.select_dtypes(include=[np.number]).columns
        columns = list(numeric_cols)
    
    normalized_df = dataframe.copy()
    min_range, max_range = feature_range
    
    for col in columns:
        if col not in normalized_df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame")
        
        if normalized_df[col].dtype in [np.float64, np.int64]:
            min_val = normalized_df[col].min()
            max_val = normalized_df[col].max()
            
            if max_val > min_val:
                normalized_df[col] = (normalized_df[col] - min_val) / (max_val - min_val)
                normalized_df[col] = normalized_df[col] * (max_range - min_range) + min_range
            else:
                normalized_df[col] = min_range
    
    return normalized_df

def detect_skewed_columns(dataframe, threshold=0.5):
    """
    Detect columns with significant skewness.
    
    Parameters:
    dataframe (pd.DataFrame): Input DataFrame
    threshold (float): Absolute skewness threshold
    
    Returns:
    dict: Dictionary with column names and their skewness values
    """
    skewed_cols = {}
    
    for col in dataframe.select_dtypes(include=[np.number]).columns:
        skewness = stats.skew(dataframe[col].dropna())
        if abs(skewness) > threshold:
            skewed_cols[col] = skewness
    
    return skewed_cols

def log_transform(dataframe, columns):
    """
    Apply log transformation to reduce skewness.
    
    Parameters:
    dataframe (pd.DataFrame): Input DataFrame
    columns (list): List of column names to transform
    
    Returns:
    pd.DataFrame: DataFrame with transformed columns
    """
    transformed_df = dataframe.copy()
    
    for col in columns:
        if col not in transformed_df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame")
        
        if transformed_df[col].min() <= 0:
            transformed_df[col] = np.log1p(transformed_df[col] - transformed_df[col].min() + 1)
        else:
            transformed_df[col] = np.log(transformed_df[col])
    
    return transformed_df

def clean_dataset(dataframe, outlier_columns=None, normalize_columns=None, 
                  normalization_method='zscore', skew_threshold=0.5):
    """
    Comprehensive data cleaning pipeline.
    
    Parameters:
    dataframe (pd.DataFrame): Input DataFrame
    outlier_columns (list): Columns for outlier removal
    normalize_columns (list): Columns for normalization
    normalization_method (str): 'zscore' or 'minmax'
    skew_threshold (float): Skewness detection threshold
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    cleaned_df = dataframe.copy()
    
    if outlier_columns:
        for col in outlier_columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
    
    skewed_cols = detect_skewed_columns(cleaned_df, skew_threshold)
    if skewed_cols:
        cleaned_df = log_transform(cleaned_df, list(skewed_cols.keys()))
    
    if normalize_columns:
        if normalization_method == 'zscore':
            cleaned_df = z_score_normalize(cleaned_df, normalize_columns)
        elif normalization_method == 'minmax':
            cleaned_df = min_max_normalize(cleaned_df, normalize_columns)
        else:
            raise ValueError("Normalization method must be 'zscore' or 'minmax'")
    
    return cleaned_df
import numpy as np

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a specified column using the IQR method.
    
    Parameters:
    data (pd.DataFrame): The input DataFrame.
    column (str): The column name to process.
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed.
    """
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data

def calculate_basic_stats(data, column):
    """
    Calculate basic statistics for a column.
    
    Parameters:
    data (pd.DataFrame): The input DataFrame.
    column (str): The column name.
    
    Returns:
    dict: Dictionary containing mean, median, and standard deviation.
    """
    stats = {
        'mean': data[column].mean(),
        'median': data[column].median(),
        'std': data[column].std()
    }
    return stats