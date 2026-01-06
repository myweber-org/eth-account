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
    if max_val - min_val == 0:
        return df[column]
    return (df[column] - min_val) / (max_val - min_val)

def standardize_zscore(df, column):
    mean_val = df[column].mean()
    std_val = df[column].std()
    if std_val == 0:
        return df[column]
    return (df[column] - mean_val) / std_val

def clean_dataset(df, numeric_columns):
    cleaned_df = df.copy()
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
    return cleaned_df

def process_features(df, numeric_columns, method='normalize'):
    processed_df = df.copy()
    for col in numeric_columns:
        if col in processed_df.columns:
            if method == 'normalize':
                processed_df[col] = normalize_minmax(processed_df, col)
            elif method == 'standardize':
                processed_df[col] = standardize_zscore(processed_df, col)
    return processed_df

if __name__ == "__main__":
    sample_data = {'feature1': [1, 2, 3, 4, 100],
                   'feature2': [10, 20, 30, 40, 50]}
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    
    cleaned = clean_dataset(df, ['feature1', 'feature2'])
    print("\nCleaned DataFrame (outliers removed):")
    print(cleaned)
    
    normalized = process_features(cleaned, ['feature1', 'feature2'], 'normalize')
    print("\nNormalized DataFrame:")
    print(normalized)import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column, threshold=1.5):
    """
    Remove outliers using IQR method
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data

def remove_outliers_zscore(data, column, threshold=3):
    """
    Remove outliers using Z-score method
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    z_scores = np.abs(stats.zscore(data[column].dropna()))
    filtered_indices = np.where(z_scores < threshold)[0]
    filtered_data = data.iloc[filtered_indices]
    return filtered_data

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

def normalize_zscore(data, column):
    """
    Normalize data using Z-score standardization
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    mean_val = data[column].mean()
    std_val = data[column].std()
    
    if std_val == 0:
        return data[column].apply(lambda x: 0)
    
    standardized = (data[column] - mean_val) / std_val
    return standardized

def clean_dataset(df, numeric_columns=None, outlier_method='iqr', normalize_method='minmax'):
    """
    Comprehensive data cleaning pipeline
    """
    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_df = df.copy()
    
    for col in numeric_columns:
        if col not in df.columns:
            continue
            
        if outlier_method == 'iqr':
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
        elif outlier_method == 'zscore':
            cleaned_df = remove_outliers_zscore(cleaned_df, col)
        
        if normalize_method == 'minmax':
            cleaned_df[col] = normalize_minmax(cleaned_df, col)
        elif normalize_method == 'zscore':
            cleaned_df[col] = normalize_zscore(cleaned_df, col)
    
    return cleaned_df

def validate_data(df, required_columns=None, allow_nan=False):
    """
    Validate data structure and content
    """
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
    
    if not allow_nan:
        nan_count = df.isnull().sum().sum()
        if nan_count > 0:
            raise ValueError(f"Data contains {nan_count} NaN values")
    
    return Trueimport pandas as pd
import numpy as np
from typing import Union, List, Optional

def remove_duplicates(df: pd.DataFrame, subset: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Remove duplicate rows from DataFrame.
    
    Args:
        df: Input DataFrame
        subset: Columns to consider for identifying duplicates
    
    Returns:
        DataFrame with duplicates removed
    """
    return df.drop_duplicates(subset=subset, keep='first')

def handle_missing_values(df: pd.DataFrame, 
                         strategy: str = 'drop', 
                         fill_value: Union[int, float, str] = None) -> pd.DataFrame:
    """
    Handle missing values in DataFrame.
    
    Args:
        df: Input DataFrame
        strategy: 'drop' to remove rows, 'fill' to fill values
        fill_value: Value to fill when strategy is 'fill'
    
    Returns:
        DataFrame with handled missing values
    """
    if strategy == 'drop':
        return df.dropna()
    elif strategy == 'fill':
        if fill_value is None:
            raise ValueError("fill_value must be provided when strategy is 'fill'")
        return df.fillna(fill_value)
    else:
        raise ValueError("strategy must be either 'drop' or 'fill'")

def normalize_column(df: pd.DataFrame, 
                    column: str, 
                    method: str = 'minmax') -> pd.DataFrame:
    """
    Normalize specified column in DataFrame.
    
    Args:
        df: Input DataFrame
        column: Column name to normalize
        method: Normalization method ('minmax' or 'zscore')
    
    Returns:
        DataFrame with normalized column
    """
    df_copy = df.copy()
    
    if method == 'minmax':
        min_val = df_copy[column].min()
        max_val = df_copy[column].max()
        if max_val != min_val:
            df_copy[column] = (df_copy[column] - min_val) / (max_val - min_val)
    
    elif method == 'zscore':
        mean_val = df_copy[column].mean()
        std_val = df_copy[column].std()
        if std_val != 0:
            df_copy[column] = (df_copy[column] - mean_val) / std_val
    
    else:
        raise ValueError("method must be either 'minmax' or 'zscore'")
    
    return df_copy

def filter_outliers(df: pd.DataFrame, 
                   column: str, 
                   method: str = 'iqr',
                   threshold: float = 1.5) -> pd.DataFrame:
    """
    Filter outliers from specified column.
    
    Args:
        df: Input DataFrame
        column: Column name to filter
        method: Outlier detection method ('iqr' or 'zscore')
        threshold: Threshold for outlier detection
    
    Returns:
        DataFrame with outliers removed
    """
    df_copy = df.copy()
    
    if method == 'iqr':
        Q1 = df_copy[column].quantile(0.25)
        Q3 = df_copy[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        mask = (df_copy[column] >= lower_bound) & (df_copy[column] <= upper_bound)
    
    elif method == 'zscore':
        z_scores = np.abs((df_copy[column] - df_copy[column].mean()) / df_copy[column].std())
        mask = z_scores <= threshold
    
    else:
        raise ValueError("method must be either 'iqr' or 'zscore'")
    
    return df_copy[mask]

def convert_data_types(df: pd.DataFrame, 
                      type_mapping: dict) -> pd.DataFrame:
    """
    Convert columns to specified data types.
    
    Args:
        df: Input DataFrame
        type_mapping: Dictionary mapping column names to target types
    
    Returns:
        DataFrame with converted data types
    """
    df_copy = df.copy()
    
    for column, dtype in type_mapping.items():
        if column in df_copy.columns:
            try:
                df_copy[column] = df_copy[column].astype(dtype)
            except ValueError as e:
                print(f"Warning: Could not convert column '{column}' to {dtype}: {e}")
    
    return df_copy

def clean_dataset(df: pd.DataFrame,
                 drop_duplicates: bool = True,
                 handle_na: str = 'drop',
                 normalize_cols: Optional[List[str]] = None,
                 filter_outlier_cols: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Comprehensive dataset cleaning pipeline.
    
    Args:
        df: Input DataFrame
        drop_duplicates: Whether to remove duplicates
        handle_na: Strategy for handling missing values
        normalize_cols: Columns to normalize
        filter_outlier_cols: Columns to filter outliers from
    
    Returns:
        Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        cleaned_df = remove_duplicates(cleaned_df)
    
    if handle_na:
        cleaned_df = handle_missing_values(cleaned_df, strategy=handle_na)
    
    if normalize_cols:
        for col in normalize_cols:
            if col in cleaned_df.columns:
                cleaned_df = normalize_column(cleaned_df, col)
    
    if filter_outlier_cols:
        for col in filter_outlier_cols:
            if col in cleaned_df.columns:
                cleaned_df = filter_outliers(cleaned_df, col)
    
    return cleaned_dfimport numpy as np
import pandas as pd

class DataCleaner:
    def __init__(self, data):
        self.data = data
        self.cleaned_data = None
        
    def detect_outliers_iqr(self, column):
        Q1 = self.data[column].quantile(0.25)
        Q3 = self.data[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = self.data[(self.data[column] < lower_bound) | (self.data[column] > upper_bound)]
        return outliers
    
    def remove_outliers(self, columns):
        clean_data = self.data.copy()
        for col in columns:
            Q1 = clean_data[col].quantile(0.25)
            Q3 = clean_data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            clean_data = clean_data[(clean_data[col] >= lower_bound) & (clean_data[col] <= upper_bound)]
        self.cleaned_data = clean_data
        return clean_data
    
    def normalize_minmax(self, columns):
        if self.cleaned_data is None:
            data_to_normalize = self.data.copy()
        else:
            data_to_normalize = self.cleaned_data.copy()
        
        for col in columns:
            min_val = data_to_normalize[col].min()
            max_val = data_to_normalize[col].max()
            if max_val != min_val:
                data_to_normalize[col] = (data_to_normalize[col] - min_val) / (max_val - min_val)
        return data_to_normalize
    
    def fill_missing_mean(self, columns):
        if self.cleaned_data is None:
            data_to_fill = self.data.copy()
        else:
            data_to_fill = self.cleaned_data.copy()
        
        for col in columns:
            if data_to_fill[col].isnull().any():
                mean_val = data_to_fill[col].mean()
                data_to_fill[col].fillna(mean_val, inplace=True)
        return data_to_fill
    
    def get_summary(self):
        summary = {
            'original_rows': len(self.data),
            'original_columns': len(self.data.columns),
            'cleaned_rows': len(self.cleaned_data) if self.cleaned_data is not None else len(self.data),
            'cleaned_columns': len(self.cleaned_data.columns) if self.cleaned_data is not None else len(self.data.columns),
            'missing_values': self.data.isnull().sum().to_dict() if self.cleaned_data is None else self.cleaned_data.isnull().sum().to_dict()
        }
        return summary

def create_sample_data():
    np.random.seed(42)
    data = {
        'feature_a': np.random.normal(100, 15, 100),
        'feature_b': np.random.exponential(50, 100),
        'feature_c': np.random.uniform(0, 1, 100)
    }
    df = pd.DataFrame(data)
    df.loc[np.random.choice(100, 5), 'feature_a'] = np.nan
    df.loc[10:15, 'feature_b'] = df['feature_b'].max() * 3
    return df

if __name__ == "__main__":
    sample_data = create_sample_data()
    cleaner = DataCleaner(sample_data)
    
    outliers = cleaner.detect_outliers_iqr('feature_b')
    print(f"Detected outliers in feature_b: {len(outliers)}")
    
    cleaned = cleaner.remove_outliers(['feature_b'])
    print(f"Data after removing outliers: {len(cleaned)} rows")
    
    normalized = cleaner.normalize_minmax(['feature_a', 'feature_b', 'feature_c'])
    print("Normalized data sample:")
    print(normalized.head())
    
    filled = cleaner.fill_missing_mean(['feature_a'])
    print("Data after filling missing values:")
    print(filled.isnull().sum())
    
    summary = cleaner.get_summary()
    print("Data cleaning summary:")
    for key, value in summary.items():
        print(f"{key}: {value}")