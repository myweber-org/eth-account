
import pandas as pd

def remove_duplicates(dataframe, subset=None, keep='first'):
    """
    Remove duplicate rows from a pandas DataFrame.
    
    Args:
        dataframe: Input pandas DataFrame.
        subset: Column label or sequence of labels to consider for identifying duplicates.
                If None, all columns are used.
        keep: Determines which duplicates to mark.
              'first' : Mark duplicates as True except for the first occurrence.
              'last' : Mark duplicates as True except for the last occurrence.
              False : Mark all duplicates as True.
    
    Returns:
        DataFrame with duplicates removed.
    """
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    
    cleaned_df = dataframe.drop_duplicates(subset=subset, keep=keep)
    return cleaned_df

def clean_numeric_columns(dataframe, columns):
    """
    Clean numeric columns by converting to appropriate type and handling errors.
    
    Args:
        dataframe: Input pandas DataFrame.
        columns: List of column names to clean.
    
    Returns:
        DataFrame with cleaned numeric columns.
    """
    for col in columns:
        if col in dataframe.columns:
            dataframe[col] = pd.to_numeric(dataframe[col], errors='coerce')
    return dataframe

def validate_dataframe(dataframe, required_columns):
    """
    Validate that DataFrame contains all required columns.
    
    Args:
        dataframe: Input pandas DataFrame.
        required_columns: List of required column names.
    
    Returns:
        Boolean indicating if all required columns are present.
    """
    missing_columns = [col for col in required_columns if col not in dataframe.columns]
    
    if missing_columns:
        print(f"Missing columns: {missing_columns}")
        return False
    return True
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
            
        df_clean = self.df.copy()
        for col in columns:
            if col in self.df.columns and self.df[col].dtype in [np.float64, np.int64]:
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - factor * IQR
                upper_bound = Q3 + factor * IQR
                
                mask = (self.df[col] >= lower_bound) & (self.df[col] <= upper_bound)
                df_clean = df_clean[mask]
        
        self.df = df_clean.reset_index(drop=True)
        removed_count = self.original_shape[0] - self.df.shape[0]
        return removed_count
    
    def normalize_minmax(self, columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
            
        df_normalized = self.df.copy()
        for col in columns:
            if col in self.df.columns and self.df[col].dtype in [np.float64, np.int64]:
                min_val = self.df[col].min()
                max_val = self.df[col].max()
                if max_val != min_val:
                    df_normalized[col] = (self.df[col] - min_val) / (max_val - min_val)
        
        self.df = df_normalized
        return self.df
    
    def standardize_zscore(self, columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
            
        df_standardized = self.df.copy()
        for col in columns:
            if col in self.df.columns and self.df[col].dtype in [np.float64, np.int64]:
                mean_val = self.df[col].mean()
                std_val = self.df[col].std()
                if std_val > 0:
                    df_standardized[col] = (self.df[col] - mean_val) / std_val
        
        self.df = df_standardized
        return self.df
    
    def fill_missing_median(self, columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
            
        df_filled = self.df.copy()
        for col in columns:
            if col in self.df.columns and self.df[col].dtype in [np.float64, np.int64]:
                median_val = self.df[col].median()
                df_filled[col] = self.df[col].fillna(median_val)
        
        self.df = df_filled
        return self.df
    
    def get_clean_data(self):
        return self.df
    
    def get_summary(self):
        summary = {
            'original_rows': self.original_shape[0],
            'current_rows': self.df.shape[0],
            'columns': self.df.shape[1],
            'removed_rows': self.original_shape[0] - self.df.shape[0],
            'numeric_columns': list(self.df.select_dtypes(include=[np.number]).columns)
        }
        return summary

def create_sample_data():
    np.random.seed(42)
    data = {
        'feature_a': np.random.normal(100, 15, 1000),
        'feature_b': np.random.exponential(50, 1000),
        'feature_c': np.random.uniform(0, 200, 1000),
        'category': np.random.choice(['A', 'B', 'C'], 1000)
    }
    
    df = pd.DataFrame(data)
    df.loc[np.random.choice(df.index, 50), 'feature_a'] = np.nan
    df.loc[np.random.choice(df.index, 20), 'feature_b'] = 9999
    
    return df

if __name__ == "__main__":
    sample_df = create_sample_data()
    cleaner = DataCleaner(sample_df)
    
    print("Initial data shape:", cleaner.get_summary()['original_rows'])
    
    removed = cleaner.remove_outliers_iqr(['feature_a', 'feature_b', 'feature_c'])
    print(f"Removed {removed} outliers")
    
    cleaner.fill_missing_median()
    cleaner.normalize_minmax()
    
    clean_df = cleaner.get_clean_data()
    print("Cleaned data shape:", clean_df.shape)
    print("\nFirst 5 rows of cleaned data:")
    print(clean_df.head())