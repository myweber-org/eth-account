import pandas as pd

def remove_duplicates(df, subset=None, keep='first'):
    """
    Remove duplicate rows from a DataFrame.
    
    Args:
        df: pandas DataFrame
        subset: column label or sequence of labels to consider for duplicates
        keep: determines which duplicates to keep ('first', 'last', False)
    
    Returns:
        DataFrame with duplicates removed
    """
    if df.empty:
        return df
    
    cleaned_df = df.drop_duplicates(subset=subset, keep=keep)
    
    removed_count = len(df) - len(cleaned_df)
    if removed_count > 0:
        print(f"Removed {removed_count} duplicate row(s)")
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Args:
        df: pandas DataFrame to validate
        required_columns: list of column names that must be present
    
    Returns:
        Boolean indicating if validation passed
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    
    if df.empty:
        print("Warning: DataFrame is empty")
        return True
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
    
    return True

def clean_numeric_columns(df, columns):
    """
    Clean numeric columns by converting to appropriate types and handling errors.
    
    Args:
        df: pandas DataFrame
        columns: list of column names to clean
    
    Returns:
        DataFrame with cleaned numeric columns
    """
    cleaned_df = df.copy()
    
    for col in columns:
        if col in cleaned_df.columns:
            cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='coerce')
    
    return cleaned_df

def get_data_summary(df):
    """
    Generate summary statistics for a DataFrame.
    
    Args:
        df: pandas DataFrame
    
    Returns:
        Dictionary containing summary statistics
    """
    summary = {
        'rows': len(df),
        'columns': len(df.columns),
        'missing_values': df.isnull().sum().sum(),
        'duplicates': df.duplicated().sum(),
        'column_types': df.dtypes.to_dict()
    }
    
    numeric_cols = df.select_dtypes(include=['number']).columns
    if len(numeric_cols) > 0:
        summary['numeric_summary'] = df[numeric_cols].describe().to_dict()
    
    return summary
import numpy as np
import pandas as pd

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
    
    return filtered_df.reset_index(drop=True)

def calculate_summary_statistics(df, column):
    """
    Calculate summary statistics for a column after outlier removal.
    
    Parameters:
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

def example_usage():
    """
    Example usage of the data cleaning functions.
    """
    np.random.seed(42)
    
    data = {
        'values': np.concatenate([
            np.random.normal(100, 15, 95),
            np.random.normal(300, 50, 5)
        ])
    }
    
    df = pd.DataFrame(data)
    
    print("Original data shape:", df.shape)
    print("Original statistics:", calculate_summary_statistics(df, 'values'))
    
    cleaned_df = remove_outliers_iqr(df, 'values')
    
    print("\nCleaned data shape:", cleaned_df.shape)
    print("Cleaned statistics:", calculate_summary_statistics(cleaned_df, 'values'))
    
    return cleaned_df

if __name__ == "__main__":
    cleaned_data = example_usage()import pandas as pd
import numpy as np
from typing import Optional

class DataCleaner:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.original_shape = df.shape
        
    def remove_duplicates(self, subset: Optional[list] = None) -> 'DataCleaner':
        self.df = self.df.drop_duplicates(subset=subset, keep='first')
        return self
        
    def fill_missing_numeric(self, strategy: str = 'mean', fill_value: Optional[float] = None) -> 'DataCleaner':
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if self.df[col].isnull().any():
                if strategy == 'mean':
                    fill_val = self.df[col].mean()
                elif strategy == 'median':
                    fill_val = self.df[col].median()
                elif strategy == 'mode':
                    fill_val = self.df[col].mode()[0]
                elif strategy == 'constant' and fill_value is not None:
                    fill_val = fill_value
                else:
                    continue
                    
                self.df[col] = self.df[col].fillna(fill_val)
        
        return self
        
    def fill_missing_categorical(self, strategy: str = 'mode', fill_value: Optional[str] = None) -> 'DataCleaner':
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns
        
        for col in categorical_cols:
            if self.df[col].isnull().any():
                if strategy == 'mode':
                    fill_val = self.df[col].mode()[0] if not self.df[col].mode().empty else 'Unknown'
                elif strategy == 'constant' and fill_value is not None:
                    fill_val = fill_value
                else:
                    continue
                    
                self.df[col] = self.df[col].fillna(fill_val)
        
        return self
        
    def remove_outliers_iqr(self, columns: list, multiplier: float = 1.5) -> 'DataCleaner':
        for col in columns:
            if col in self.df.columns and self.df[col].dtype in [np.float64, np.int64]:
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - multiplier * IQR
                upper_bound = Q3 + multiplier * IQR
                
                self.df = self.df[(self.df[col] >= lower_bound) & (self.df[col] <= upper_bound)]
        
        return self
        
    def standardize_numeric(self, columns: list) -> 'DataCleaner':
        for col in columns:
            if col in self.df.columns and self.df[col].dtype in [np.float64, np.int64]:
                mean = self.df[col].mean()
                std = self.df[col].std()
                if std > 0:
                    self.df[col] = (self.df[col] - mean) / std
        
        return self
        
    def get_cleaned_data(self) -> pd.DataFrame:
        return self.df
        
    def get_summary(self) -> dict:
        return {
            'original_rows': self.original_shape[0],
            'original_columns': self.original_shape[1],
            'cleaned_rows': self.df.shape[0],
            'cleaned_columns': self.df.shape[1],
            'rows_removed': self.original_shape[0] - self.df.shape[0],
            'missing_values_before': self.original_shape[0] * self.original_shape[1] - self.df.count().sum(),
            'missing_values_after': self.df.isnull().sum().sum()
        }

def clean_csv_file(input_path: str, output_path: str, **kwargs) -> dict:
    df = pd.read_csv(input_path)
    cleaner = DataCleaner(df)
    
    if 'remove_dups' in kwargs and kwargs['remove_dups']:
        cleaner.remove_duplicates(kwargs.get('dup_subset'))
    
    if 'fill_numeric' in kwargs:
        cleaner.fill_missing_numeric(kwargs['fill_numeric'], kwargs.get('numeric_fill_value'))
    
    if 'fill_categorical' in kwargs:
        cleaner.fill_missing_categorical(kwargs['fill_categorical'], kwargs.get('categorical_fill_value'))
    
    if 'remove_outliers' in kwargs and kwargs['remove_outliers']:
        cleaner.remove_outliers_iqr(kwargs.get('outlier_columns', []), kwargs.get('outlier_multiplier', 1.5))
    
    if 'standardize' in kwargs and kwargs['standardize']:
        cleaner.standardize_numeric(kwargs.get('standardize_columns', []))
    
    cleaned_df = cleaner.get_cleaned_data()
    cleaned_df.to_csv(output_path, index=False)
    
    return cleaner.get_summary()