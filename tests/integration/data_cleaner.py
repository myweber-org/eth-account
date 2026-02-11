
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range method.
    
    Parameters:
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

def clean_numeric_data(df, columns=None):
    """
    Clean numeric data by removing outliers from specified columns.
    If no columns specified, clean all numeric columns.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    columns (list): List of column names to clean
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    if columns is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        columns = list(numeric_cols)
    
    cleaned_df = df.copy()
    
    for col in columns:
        if col in cleaned_df.columns:
            try:
                cleaned_df = remove_outliers_iqr(cleaned_df, col)
            except Exception as e:
                print(f"Warning: Could not clean column '{col}': {e}")
    
    return cleaned_df

def get_data_summary(df):
    """
    Generate summary statistics for numeric columns.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    
    Returns:
    pd.DataFrame: Summary statistics
    """
    return df.describe()

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'A': np.random.normal(100, 15, 1000),
        'B': np.random.exponential(50, 1000),
        'C': np.random.uniform(0, 200, 1000)
    }
    
    df = pd.DataFrame(sample_data)
    print("Original data shape:", df.shape)
    print("\nOriginal summary:")
    print(get_data_summary(df))
    
    cleaned_df = clean_numeric_data(df)
    print("\nCleaned data shape:", cleaned_df.shape)
    print("\nCleaned summary:")
    print(get_data_summary(cleaned_df))
import numpy as np

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a dataset using the Interquartile Range (IQR) method.
    
    Parameters:
    data (numpy.ndarray): Input data array.
    column (int): Column index to process for outlier removal.
    
    Returns:
    numpy.ndarray: Data with outliers removed.
    """
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a numpy array")
    
    if column >= data.shape[1]:
        raise IndexError("Column index out of bounds")
    
    column_data = data[:, column]
    
    Q1 = np.percentile(column_data, 25)
    Q3 = np.percentile(column_data, 75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    mask = (column_data >= lower_bound) & (column_data <= upper_bound)
    
    return data[mask]

def calculate_statistics(data, column):
    """
    Calculate basic statistics for a data column.
    
    Parameters:
    data (numpy.ndarray): Input data array.
    column (int): Column index to analyze.
    
    Returns:
    dict: Dictionary containing mean, median, and standard deviation.
    """
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a numpy array")
    
    column_data = data[:, column]
    
    stats = {
        'mean': np.mean(column_data),
        'median': np.median(column_data),
        'std': np.std(column_data),
        'min': np.min(column_data),
        'max': np.max(column_data)
    }
    
    return stats

def clean_dataset(data, columns_to_clean=None):
    """
    Clean dataset by removing outliers from specified columns.
    
    Parameters:
    data (numpy.ndarray): Input data array.
    columns_to_clean (list): List of column indices to clean. If None, clean all columns.
    
    Returns:
    numpy.ndarray: Cleaned data array.
    """
    if columns_to_clean is None:
        columns_to_clean = list(range(data.shape[1]))
    
    cleaned_data = data.copy()
    
    for column in columns_to_clean:
        if column < data.shape[1]:
            cleaned_data = remove_outliers_iqr(cleaned_data, column)
    
    return cleaned_data

if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    sample_data = np.random.randn(100, 3)
    sample_data[0, 0] = 10  # Add an outlier
    
    print("Original data shape:", sample_data.shape)
    print("First few rows of original data:")
    print(sample_data[:5])
    
    cleaned = clean_dataset(sample_data, columns_to_clean=[0])
    
    print("\nCleaned data shape:", cleaned.shape)
    print("First few rows of cleaned data:")
    print(cleaned[:5])
    
    stats = calculate_statistics(sample_data, 0)
    print("\nStatistics for column 0 (original):")
    for key, value in stats.items():
        print(f"{key}: {value:.4f}")
import pandas as pd
import re

def clean_text_column(df, column_name):
    """
    Normalize text by converting to lowercase and removing extra whitespace.
    """
    if column_name in df.columns:
        df[column_name] = df[column_name].astype(str).str.lower()
        df[column_name] = df[column_name].apply(lambda x: re.sub(r'\s+', ' ', x).strip())
    return df

def remove_duplicates(df, subset=None):
    """
    Remove duplicate rows from the DataFrame.
    """
    return df.drop_duplicates(subset=subset, keep='first')

def process_dataframe(input_file, output_file):
    """
    Main function to load, clean, and save the DataFrame.
    """
    try:
        df = pd.read_csv(input_file)
        print(f"Loaded data from {input_file}")
        
        df = remove_duplicates(df)
        print("Removed duplicate rows")
        
        text_columns = [col for col in df.columns if df[col].dtype == 'object']
        for col in text_columns:
            df = clean_text_column(df, col)
        print(f"Cleaned text columns: {text_columns}")
        
        df.to_csv(output_file, index=False)
        print(f"Saved cleaned data to {output_file}")
        
    except FileNotFoundError:
        print(f"Error: File {input_file} not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    process_dataframe('raw_data.csv', 'cleaned_data.csv')
def remove_duplicates(seq):
    seen = set()
    result = []
    for item in seq:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result
import pandas as pd
import numpy as np
from typing import Optional, Union, List

class DataCleaner:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.original_shape = df.shape
        
    def remove_duplicates(self, subset: Optional[List[str]] = None) -> pd.DataFrame:
        self.df = self.df.drop_duplicates(subset=subset, keep='first')
        return self.df
    
    def handle_missing_values(self, 
                             strategy: str = 'mean', 
                             columns: Optional[List[str]] = None) -> pd.DataFrame:
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        for col in columns:
            if col in self.df.columns:
                if strategy == 'mean':
                    fill_value = self.df[col].mean()
                elif strategy == 'median':
                    fill_value = self.df[col].median()
                elif strategy == 'mode':
                    fill_value = self.df[col].mode()[0]
                elif strategy == 'constant':
                    fill_value = 0
                else:
                    raise ValueError(f"Unknown strategy: {strategy}")
                
                self.df[col] = self.df[col].fillna(fill_value)
        
        return self.df
    
    def remove_outliers_iqr(self, 
                           columns: Optional[List[str]] = None, 
                           multiplier: float = 1.5) -> pd.DataFrame:
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        for col in columns:
            if col in self.df.columns:
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - multiplier * IQR
                upper_bound = Q3 + multiplier * IQR
                
                self.df = self.df[(self.df[col] >= lower_bound) & 
                                 (self.df[col] <= upper_bound)]
        
        return self.df
    
    def normalize_column(self, 
                        column: str, 
                        method: str = 'minmax') -> pd.DataFrame:
        if column not in self.df.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame")
        
        if method == 'minmax':
            min_val = self.df[column].min()
            max_val = self.df[column].max()
            if max_val != min_val:
                self.df[column] = (self.df[column] - min_val) / (max_val - min_val)
        
        elif method == 'zscore':
            mean_val = self.df[column].mean()
            std_val = self.df[column].std()
            if std_val != 0:
                self.df[column] = (self.df[column] - mean_val) / std_val
        
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        return self.df
    
    def get_cleaning_report(self) -> dict:
        removed_rows = self.original_shape[0] - self.df.shape[0]
        removed_cols = self.original_shape[1] - self.df.shape[1]
        
        missing_values = self.df.isnull().sum().sum()
        duplicate_rows = self.df.duplicated().sum()
        
        return {
            'original_shape': self.original_shape,
            'current_shape': self.df.shape,
            'rows_removed': removed_rows,
            'columns_removed': removed_cols,
            'remaining_missing_values': missing_values,
            'remaining_duplicates': duplicate_rows
        }
    
    def save_cleaned_data(self, filepath: str, format: str = 'csv'):
        if format == 'csv':
            self.df.to_csv(filepath, index=False)
        elif format == 'excel':
            self.df.to_excel(filepath, index=False)
        elif format == 'parquet':
            self.df.to_parquet(filepath, index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")

def load_and_clean_csv(filepath: str, 
                      missing_strategy: str = 'mean',
                      remove_outliers: bool = True) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    cleaner = DataCleaner(df)
    
    cleaner.remove_duplicates()
    cleaner.handle_missing_values(strategy=missing_strategy)
    
    if remove_outliers:
        cleaner.remove_outliers_iqr()
    
    report = cleaner.get_cleaning_report()
    print(f"Data cleaning completed. Removed {report['rows_removed']} rows.")
    
    return cleaner.df