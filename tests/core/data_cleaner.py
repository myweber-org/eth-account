
import pandas as pd
import re

def clean_dataframe(df, text_columns=None, drop_duplicates=True):
    """
    Clean a pandas DataFrame by removing duplicates and standardizing text columns.
    
    Args:
        df: pandas DataFrame to clean
        text_columns: list of column names to standardize (lowercase, strip whitespace)
        drop_duplicates: whether to remove duplicate rows
    
    Returns:
        Cleaned pandas DataFrame
    """
    df_clean = df.copy()
    
    if drop_duplicates:
        initial_rows = len(df_clean)
        df_clean = df_clean.drop_duplicates()
        removed = initial_rows - len(df_clean)
        print(f"Removed {removed} duplicate rows")
    
    if text_columns:
        for col in text_columns:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].astype(str).str.lower().str.strip()
                print(f"Standardized text in column: {col}")
    
    return df_clean

def validate_email(email):
    """
    Validate email format using regex pattern.
    
    Args:
        email: string email address to validate
    
    Returns:
        Boolean indicating if email is valid
    """
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, str(email)))

def extract_numeric(text):
    """
    Extract numeric values from text string.
    
    Args:
        text: string containing numeric values
    
    Returns:
        List of numeric values found in text
    """
    numbers = re.findall(r'\d+\.?\d*', str(text))
    return [float(num) if '.' in num else int(num) for num in numbers]
import pandas as pd
import numpy as np
from scipy import stats

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.original_columns = df.columns.tolist()
    
    def remove_outliers_zscore(self, columns=None, threshold=3):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        df_clean = self.df.copy()
        for col in columns:
            if col in df_clean.columns:
                z_scores = np.abs(stats.zscore(df_clean[col].dropna()))
                df_clean = df_clean[(z_scores < threshold) | (df_clean[col].isna())]
        
        self.df = df_clean
        return self
    
    def normalize_minmax(self, columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        for col in columns:
            if col in self.df.columns:
                col_min = self.df[col].min()
                col_max = self.df[col].max()
                if col_max > col_min:
                    self.df[col] = (self.df[col] - col_min) / (col_max - col_min)
        
        return self
    
    def fill_missing_median(self, columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        for col in columns:
            if col in self.df.columns:
                median_val = self.df[col].median()
                self.df[col].fillna(median_val, inplace=True)
        
        return self
    
    def get_cleaned_data(self):
        return self.df
    
    def get_summary(self):
        summary = {
            'original_rows': len(self.df),
            'cleaned_rows': len(self.df),
            'numeric_columns': self.df.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical_columns': self.df.select_dtypes(include=['object']).columns.tolist(),
            'missing_values': self.df.isnull().sum().to_dict()
        }
        return summary

def process_dataset(file_path):
    try:
        df = pd.read_csv(file_path)
        cleaner = DataCleaner(df)
        
        cleaner.fill_missing_median()
        cleaner.remove_outliers_zscore()
        cleaner.normalize_minmax()
        
        cleaned_df = cleaner.get_cleaned_data()
        summary = cleaner.get_summary()
        
        return cleaned_df, summary
    except Exception as e:
        print(f"Error processing dataset: {e}")
        return None, None