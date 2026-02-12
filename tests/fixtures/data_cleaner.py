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
    df[column + '_normalized'] = (df[column] - min_val) / (max_val - min_val)
    return df

def clean_dataset(df, numeric_columns):
    cleaned_df = df.copy()
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
            cleaned_df = normalize_minmax(cleaned_df, col)
    return cleaned_df

def export_cleaned_data(df, output_path):
    df.to_csv(output_path, index=False)
    print(f"Cleaned data exported to {output_path}")
import pandas as pd
import numpy as np
from scipy import stats

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
    def detect_outliers_iqr(self, column, threshold=1.5):
        if column not in self.numeric_columns:
            raise ValueError(f"Column {column} is not numeric")
            
        Q1 = self.df[column].quantile(0.25)
        Q3 = self.df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        
        outliers = self.df[(self.df[column] < lower_bound) | (self.df[column] > upper_bound)]
        return outliers.index.tolist()
    
    def remove_outliers(self, columns=None, threshold=1.5):
        if columns is None:
            columns = self.numeric_columns
            
        outlier_indices = set()
        for col in columns:
            if col in self.numeric_columns:
                indices = self.detect_outliers_iqr(col, threshold)
                outlier_indices.update(indices)
        
        clean_df = self.df.drop(index=list(outlier_indices))
        return clean_df
    
    def impute_missing_median(self, columns=None):
        if columns is None:
            columns = self.numeric_columns
            
        imputed_df = self.df.copy()
        for col in columns:
            if col in self.numeric_columns and imputed_df[col].isnull().any():
                median_val = imputed_df[col].median()
                imputed_df[col].fillna(median_val, inplace=True)
        
        return imputed_df
    
    def standardize_numeric(self, columns=None):
        if columns is None:
            columns = self.numeric_columns
            
        standardized_df = self.df.copy()
        for col in columns:
            if col in self.numeric_columns:
                mean_val = standardized_df[col].mean()
                std_val = standardized_df[col].std()
                if std_val > 0:
                    standardized_df[col] = (standardized_df[col] - mean_val) / std_val
        
        return standardized_df
    
    def get_summary(self):
        summary = {
            'original_rows': len(self.df),
            'original_columns': len(self.df.columns),
            'numeric_columns': self.numeric_columns,
            'missing_values': self.df.isnull().sum().sum(),
            'missing_percentage': (self.df.isnull().sum().sum() / (len(self.df) * len(self.df.columns))) * 100
        }
        return summary

def load_and_clean_data(filepath):
    df = pd.read_csv(filepath)
    cleaner = DataCleaner(df)
    
    print("Data Summary:")
    summary = cleaner.get_summary()
    for key, value in summary.items():
        print(f"{key}: {value}")
    
    if summary['missing_values'] > 0:
        df = cleaner.impute_missing_median()
        print(f"Imputed {summary['missing_values']} missing values")
    
    df = cleaner.remove_outliers(threshold=1.5)
    print(f"Removed outliers using IQR method")
    
    df = cleaner.standardize_numeric()
    print("Standardized numeric columns")
    
    return df

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'A': [1, 2, 3, 4, 5, 100, 7, 8, 9, 10],
        'B': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        'C': [1.1, 2.2, np.nan, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.0],
        'category': ['X', 'Y', 'X', 'Y', 'X', 'Y', 'X', 'Y', 'X', 'Y']
    })
    
    cleaner = DataCleaner(sample_data)
    print("Sample data cleaning demonstration:")
    print("Original data shape:", sample_data.shape)
    
    clean_data = cleaner.remove_outliers()
    print("After outlier removal:", clean_data.shape)
    
    imputed_data = cleaner.impute_missing_median()
    print("After missing value imputation:", imputed_data.shape)
    
    standardized_data = cleaner.standardize_numeric()
    print("After standardization - mean of column A:", standardized_data['A'].mean())
import pandas as pd
import numpy as np
from typing import List, Optional

class DataCleaner:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.original_shape = df.shape
        
    def remove_duplicates(self, subset: Optional[List[str]] = None, keep: str = 'first') -> pd.DataFrame:
        """
        Remove duplicate rows from the DataFrame.
        
        Args:
            subset: Column labels to consider for identifying duplicates.
            keep: Determines which duplicates to keep.
        
        Returns:
            Cleaned DataFrame with duplicates removed.
        """
        cleaned_df = self.df.drop_duplicates(subset=subset, keep=keep)
        removed_count = len(self.df) - len(cleaned_df)
        
        if removed_count > 0:
            print(f"Removed {removed_count} duplicate rows.")
            print(f"Original shape: {self.original_shape}")
            print(f"New shape: {cleaned_df.shape}")
        
        return cleaned_df
    
    def fill_missing_values(self, column: str, method: str = 'mean') -> pd.DataFrame:
        """
        Fill missing values in a specified column.
        
        Args:
            column: Name of the column to fill.
            method: Method to use for filling ('mean', 'median', 'mode', or 'constant').
        
        Returns:
            DataFrame with filled missing values.
        """
        if column not in self.df.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame.")
        
        if method == 'mean':
            fill_value = self.df[column].mean()
        elif method == 'median':
            fill_value = self.df[column].median()
        elif method == 'mode':
            fill_value = self.df[column].mode()[0]
        elif method == 'constant':
            fill_value = 0
        else:
            raise ValueError("Method must be 'mean', 'median', 'mode', or 'constant'.")
        
        missing_count = self.df[column].isnull().sum()
        self.df[column] = self.df[column].fillna(fill_value)
        
        if missing_count > 0:
            print(f"Filled {missing_count} missing values in column '{column}' using {method} method.")
        
        return self.df
    
    def remove_outliers(self, column: str, method: str = 'iqr', threshold: float = 1.5) -> pd.DataFrame:
        """
        Remove outliers from a specified column.
        
        Args:
            column: Name of the column to process.
            method: Method to detect outliers ('iqr' or 'zscore').
            threshold: Threshold for outlier detection.
        
        Returns:
            DataFrame with outliers removed.
        """
        if column not in self.df.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame.")
        
        original_len = len(self.df)
        
        if method == 'iqr':
            Q1 = self.df[column].quantile(0.25)
            Q3 = self.df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            mask = (self.df[column] >= lower_bound) & (self.df[column] <= upper_bound)
        
        elif method == 'zscore':
            mean = self.df[column].mean()
            std = self.df[column].std()
            z_scores = np.abs((self.df[column] - mean) / std)
            mask = z_scores <= threshold
        
        else:
            raise ValueError("Method must be 'iqr' or 'zscore'.")
        
        self.df = self.df[mask]
        removed_count = original_len - len(self.df)
        
        if removed_count > 0:
            print(f"Removed {removed_count} outliers from column '{column}' using {method} method.")
        
        return self.df
    
    def get_summary(self) -> dict:
        """
        Get summary statistics of the current DataFrame.
        
        Returns:
            Dictionary containing summary statistics.
        """
        summary = {
            'original_shape': self.original_shape,
            'current_shape': self.df.shape,
            'columns': list(self.df.columns),
            'dtypes': self.df.dtypes.to_dict(),
            'missing_values': self.df.isnull().sum().to_dict(),
            'memory_usage': self.df.memory_usage(deep=True).sum()
        }
        return summary

def clean_dataset(file_path: str, output_path: Optional[str] = None) -> pd.DataFrame:
    """
    Convenience function to clean a dataset from a file.
    
    Args:
        file_path: Path to the input data file.
        output_path: Optional path to save cleaned data.
    
    Returns:
        Cleaned DataFrame.
    """
    df = pd.read_csv(file_path)
    cleaner = DataCleaner(df)
    
    cleaner.remove_duplicates()
    
    for column in df.columns:
        if df[column].dtype in ['float64', 'int64']:
            cleaner.fill_missing_values(column, 'mean')
    
    summary = cleaner.get_summary()
    print("Data cleaning completed.")
    print(f"Original records: {summary['original_shape'][0]}")
    print(f"Final records: {summary['current_shape'][0]}")
    
    if output_path:
        cleaner.df.to_csv(output_path, index=False)
        print(f"Cleaned data saved to: {output_path}")
    
    return cleaner.df

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'id': [1, 2, 2, 3, 4, 5, 5, 6],
        'value': [10.5, 20.3, 20.3, np.nan, 50.1, 1000.0, 1000.0, 15.7],
        'category': ['A', 'B', 'B', 'C', 'D', 'E', 'E', 'F']
    })
    
    cleaner = DataCleaner(sample_data)
    cleaned = cleaner.remove_duplicates()
    cleaned = cleaner.fill_missing_values('value', 'mean')
    cleaned = cleaner.remove_outliers('value', 'iqr')
    
    print("\nFinal DataFrame:")
    print(cleaned)
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
import numpy as np
import pandas as pd

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

def calculate_summary_statistics(df, column):
    """
    Calculate summary statistics for a DataFrame column.
    
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
        'count': df[column].count(),
        'missing': df[column].isnull().sum()
    }
    
    return stats

def normalize_column(df, column, method='minmax'):
    """
    Normalize a DataFrame column using specified method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to normalize
    method (str): Normalization method ('minmax' or 'zscore')
    
    Returns:
    pd.DataFrame: DataFrame with normalized column
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
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
        raise ValueError(f"Unknown normalization method: {method}")
    
    return df_copy