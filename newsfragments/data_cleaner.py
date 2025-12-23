
import pandas as pd
import numpy as np
from scipy import stats

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.original_shape = df.shape
        
    def remove_outliers_iqr(self, columns=None, threshold=1.5):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
            
        df_clean = self.df.copy()
        for col in columns:
            if col in df_clean.columns:
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                mask = (df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)
                df_clean = df_clean[mask]
                
        self.df = df_clean
        return self
        
    def fill_missing_values(self, strategy='mean', fill_value=None):
        df_filled = self.df.copy()
        
        for col in df_filled.columns:
            if df_filled[col].isnull().any():
                if strategy == 'mean' and pd.api.types.is_numeric_dtype(df_filled[col]):
                    df_filled[col] = df_filled[col].fillna(df_filled[col].mean())
                elif strategy == 'median' and pd.api.types.is_numeric_dtype(df_filled[col]):
                    df_filled[col] = df_filled[col].fillna(df_filled[col].median())
                elif strategy == 'mode':
                    df_filled[col] = df_filled[col].fillna(df_filled[col].mode()[0])
                elif strategy == 'constant' and fill_value is not None:
                    df_filled[col] = df_filled[col].fillna(fill_value)
                elif strategy == 'ffill':
                    df_filled[col] = df_filled[col].fillna(method='ffill')
                elif strategy == 'bfill':
                    df_filled[col] = df_filled[col].fillna(method='bfill')
                    
        self.df = df_filled
        return self
        
    def remove_duplicates(self, subset=None, keep='first'):
        self.df = self.df.drop_duplicates(subset=subset, keep=keep)
        return self
        
    def normalize_data(self, columns=None, method='minmax'):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
            
        df_normalized = self.df.copy()
        
        for col in columns:
            if col in df_normalized.columns and pd.api.types.is_numeric_dtype(df_normalized[col]):
                if method == 'minmax':
                    min_val = df_normalized[col].min()
                    max_val = df_normalized[col].max()
                    if max_val != min_val:
                        df_normalized[col] = (df_normalized[col] - min_val) / (max_val - min_val)
                elif method == 'zscore':
                    mean_val = df_normalized[col].mean()
                    std_val = df_normalized[col].std()
                    if std_val > 0:
                        df_normalized[col] = (df_normalized[col] - mean_val) / std_val
                        
        self.df = df_normalized
        return self
        
    def get_cleaned_data(self):
        return self.df
        
    def get_cleaning_report(self):
        report = {
            'original_rows': self.original_shape[0],
            'original_columns': self.original_shape[1],
            'cleaned_rows': self.df.shape[0],
            'cleaned_columns': self.df.shape[1],
            'rows_removed': self.original_shape[0] - self.df.shape[0],
            'missing_values_before': self.df.isnull().sum().sum(),
            'missing_values_after': 0
        }
        return report
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
    
    return filtered_df.copy()

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

def process_numerical_data(df, columns=None):
    """
    Process multiple numerical columns by removing outliers.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    columns (list): List of column names to process. If None, processes all numerical columns.
    
    Returns:
    pd.DataFrame: Processed DataFrame with outliers removed
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    processed_df = df.copy()
    
    for column in columns:
        if column in processed_df.columns:
            try:
                processed_df = remove_outliers_iqr(processed_df, column)
            except Exception as e:
                print(f"Error processing column {column}: {e}")
    
    return processed_df

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'A': np.random.normal(100, 15, 1000),
        'B': np.random.exponential(50, 1000),
        'C': np.random.uniform(0, 200, 1000)
    }
    
    df = pd.DataFrame(sample_data)
    
    print("Original DataFrame shape:", df.shape)
    print("Original summary statistics:")
    for col in df.columns:
        stats = calculate_summary_statistics(df, col)
        print(f"\n{col}: {stats}")
    
    cleaned_df = process_numerical_data(df)
    
    print("\nCleaned DataFrame shape:", cleaned_df.shape)
    print("Cleaned summary statistics:")
    for col in cleaned_df.columns:
        stats = calculate_summary_statistics(cleaned_df, col)
        print(f"\n{col}: {stats}")