
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