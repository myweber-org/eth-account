
import numpy as np
import pandas as pd
from scipy import stats

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.original_shape = df.shape
        
    def remove_outliers_iqr(self, columns=None, threshold=1.5):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        clean_df = self.df.copy()
        for col in columns:
            if col in clean_df.columns:
                Q1 = clean_df[col].quantile(0.25)
                Q3 = clean_df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                clean_df = clean_df[(clean_df[col] >= lower_bound) & (clean_df[col] <= upper_bound)]
        
        removed_count = len(self.df) - len(clean_df)
        self.df = clean_df
        return removed_count
    
    def normalize_data(self, columns=None, method='zscore'):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        normalized_df = self.df.copy()
        for col in columns:
            if col in normalized_df.columns:
                if method == 'zscore':
                    normalized_df[col] = stats.zscore(normalized_df[col])
                elif method == 'minmax':
                    col_min = normalized_df[col].min()
                    col_max = normalized_df[col].max()
                    normalized_df[col] = (normalized_df[col] - col_min) / (col_max - col_min)
                elif method == 'robust':
                    median = normalized_df[col].median()
                    iqr = normalized_df[col].quantile(0.75) - normalized_df[col].quantile(0.25)
                    normalized_df[col] = (normalized_df[col] - median) / iqr
        
        self.df = normalized_df
        return self.df
    
    def handle_missing_values(self, strategy='mean', fill_value=None):
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        categorical_cols = self.df.select_dtypes(exclude=[np.number]).columns
        
        if strategy == 'drop':
            self.df = self.df.dropna()
        elif strategy == 'fill':
            if fill_value is not None:
                self.df = self.df.fillna(fill_value)
            else:
                for col in numeric_cols:
                    self.df[col] = self.df[col].fillna(self.df[col].mean())
                for col in categorical_cols:
                    self.df[col] = self.df[col].fillna(self.df[col].mode()[0] if not self.df[col].mode().empty else 'Unknown')
        
        return self.df
    
    def get_cleaning_report(self):
        report = {
            'original_rows': self.original_shape[0],
            'current_rows': len(self.df),
            'original_columns': self.original_shape[1],
            'current_columns': self.df.shape[1],
            'rows_removed': self.original_shape[0] - len(self.df),
            'missing_values': self.df.isnull().sum().sum()
        }
        return report
    
    def save_cleaned_data(self, filepath, format='csv'):
        if format == 'csv':
            self.df.to_csv(filepath, index=False)
        elif format == 'excel':
            self.df.to_excel(filepath, index=False)
        elif format == 'parquet':
            self.df.to_parquet(filepath, index=False)
        return filepath