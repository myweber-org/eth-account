import numpy as np
import pandas as pd

def remove_outliers_iqr(data, column, factor=1.5):
    """
    Remove outliers using the Interquartile Range method.
    
    Args:
        data: pandas DataFrame
        column: column name to process
        factor: IQR multiplier for outlier detection
    
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

def normalize_minmax(data, column):
    """
    Normalize data using Min-Max scaling.
    
    Args:
        data: pandas DataFrame
        column: column name to normalize
    
    Returns:
        DataFrame with normalized column
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    min_val = data[column].min()
    max_val = data[column].max()
    
    if max_val == min_val:
        return data
    
    data[column + '_normalized'] = (data[column] - min_val) / (max_val - min_val)
    return data

def standardize_zscore(data, column):
    """
    Standardize data using Z-score normalization.
    
    Args:
        data: pandas DataFrame
        column: column name to standardize
    
    Returns:
        DataFrame with standardized column
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    mean_val = data[column].mean()
    std_val = data[column].std()
    
    if std_val == 0:
        return data
    
    data[column + '_standardized'] = (data[column] - mean_val) / std_val
    return data

def clean_dataset(data, numeric_columns=None, outlier_factor=1.5):
    """
    Comprehensive data cleaning pipeline.
    
    Args:
        data: pandas DataFrame
        numeric_columns: list of numeric columns to process
        outlier_factor: IQR multiplier for outlier detection
    
    Returns:
        Cleaned DataFrame
    """
    if numeric_columns is None:
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_data = data.copy()
    
    for column in numeric_columns:
        if column in cleaned_data.columns:
            cleaned_data = remove_outliers_iqr(cleaned_data, column, outlier_factor)
            cleaned_data = normalize_minmax(cleaned_data, column)
            cleaned_data = standardize_zscore(cleaned_data, column)
    
    return cleaned_data

def validate_data(data, required_columns):
    """
    Validate that required columns exist and have no null values.
    
    Args:
        data: pandas DataFrame
        required_columns: list of required column names
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        return False, f"Missing required columns: {missing_columns}"
    
    null_counts = data[required_columns].isnull().sum()
    columns_with_nulls = null_counts[null_counts > 0].index.tolist()
    
    if columns_with_nulls:
        return False, f"Columns with null values: {columns_with_nulls}"
    
    return True, "Data validation passed"import pandas as pd

def clean_dataframe(df, drop_duplicates=True, fill_missing=True, fill_value=0):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        drop_duplicates (bool): Whether to drop duplicate rows.
        fill_missing (bool): Whether to fill missing values.
        fill_value: Value to use for filling missing data.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    if fill_missing:
        cleaned_df = cleaned_df.fillna(fill_value)
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate.
        required_columns (list): List of required column names.
    
    Returns:
        tuple: (is_valid, error_message)
    """
    if df.empty:
        return False, "DataFrame is empty"
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
    
    return True, "DataFrame is valid"
import pandas as pd
import numpy as np
from typing import List, Optional

class DataCleaner:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.original_shape = df.shape
        
    def remove_duplicates(self, subset: Optional[List[str]] = None) -> pd.DataFrame:
        """Remove duplicate rows from the dataframe."""
        initial_count = len(self.df)
        self.df = self.df.drop_duplicates(subset=subset, keep='first')
        removed = initial_count - len(self.df)
        print(f"Removed {removed} duplicate rows")
        return self.df
    
    def normalize_numeric(self, columns: List[str]) -> pd.DataFrame:
        """Normalize numeric columns to range [0, 1]."""
        for col in columns:
            if col in self.df.columns and pd.api.types.is_numeric_dtype(self.df[col]):
                min_val = self.df[col].min()
                max_val = self.df[col].max()
                if max_val > min_val:
                    self.df[col] = (self.df[col] - min_val) / (max_val - min_val)
                else:
                    self.df[col] = 0
        print(f"Normalized columns: {columns}")
        return self.df
    
    def fill_missing(self, strategy: str = 'mean', columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Fill missing values using specified strategy."""
        if columns is None:
            columns = self.df.columns.tolist()
        
        for col in columns:
            if col in self.df.columns:
                if strategy == 'mean' and pd.api.types.is_numeric_dtype(self.df[col]):
                    self.df[col] = self.df[col].fillna(self.df[col].mean())
                elif strategy == 'median' and pd.api.types.is_numeric_dtype(self.df[col]):
                    self.df[col] = self.df[col].fillna(self.df[col].median())
                elif strategy == 'mode':
                    self.df[col] = self.df[col].fillna(self.df[col].mode()[0] if not self.df[col].mode().empty else 0)
                elif strategy == 'zero':
                    self.df[col] = self.df[col].fillna(0)
        
        print(f"Filled missing values using '{strategy}' strategy")
        return self.df
    
    def get_cleaning_report(self) -> dict:
        """Generate a report of cleaning operations."""
        final_shape = self.df.shape
        return {
            'original_rows': self.original_shape[0],
            'original_columns': self.original_shape[1],
            'final_rows': final_shape[0],
            'final_columns': final_shape[1],
            'rows_removed': self.original_shape[0] - final_shape[0],
            'columns_removed': self.original_shape[1] - final_shape[1]
        }

def create_sample_data() -> pd.DataFrame:
    """Create sample dataframe for testing."""
    data = {
        'id': [1, 2, 3, 4, 5, 5, 6],
        'value': [10.5, 20.3, np.nan, 40.7, 50.1, 50.1, 60.9],
        'category': ['A', 'B', 'A', 'B', 'A', 'A', 'C'],
        'score': [0.8, 0.6, 0.9, 0.7, 0.5, 0.5, 0.4]
    }
    return pd.DataFrame(data)

if __name__ == "__main__":
    df = create_sample_data()
    cleaner = DataCleaner(df)
    
    cleaner.remove_duplicates(subset=['id'])
    cleaner.fill_missing(strategy='mean', columns=['value'])
    cleaner.normalize_numeric(columns=['score'])
    
    report = cleaner.get_cleaning_report()
    print("\nCleaning Report:")
    for key, value in report.items():
        print(f"{key}: {value}")
    
    print("\nCleaned Data:")
    print(cleaner.df)