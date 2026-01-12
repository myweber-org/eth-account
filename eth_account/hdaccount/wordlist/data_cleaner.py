
import pandas as pd
import numpy as np
from typing import List, Optional

class DataCleaner:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.original_shape = df.shape
        
    def remove_duplicates(self, subset: Optional[List[str]] = None, keep: str = 'first') -> 'DataCleaner':
        self.df = self.df.drop_duplicates(subset=subset, keep=keep)
        return self
        
    def handle_missing_values(self, strategy: str = 'drop', fill_value: Optional[float] = None) -> 'DataCleaner':
        if strategy == 'drop':
            self.df = self.df.dropna()
        elif strategy == 'fill':
            if fill_value is not None:
                self.df = self.df.fillna(fill_value)
            else:
                self.df = self.df.fillna(self.df.mean())
        return self
        
    def normalize_column(self, column: str, method: str = 'minmax') -> 'DataCleaner':
        if column not in self.df.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame")
            
        if method == 'minmax':
            col_min = self.df[column].min()
            col_max = self.df[column].max()
            if col_max != col_min:
                self.df[column] = (self.df[column] - col_min) / (col_max - col_min)
        elif method == 'zscore':
            col_mean = self.df[column].mean()
            col_std = self.df[column].std()
            if col_std > 0:
                self.df[column] = (self.df[column] - col_mean) / col_std
        return self
        
    def remove_outliers(self, column: str, threshold: float = 3.0) -> 'DataCleaner':
        if column not in self.df.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame")
            
        z_scores = np.abs((self.df[column] - self.df[column].mean()) / self.df[column].std())
        self.df = self.df[z_scores < threshold]
        return self
        
    def get_cleaned_data(self) -> pd.DataFrame:
        return self.df.copy()
        
    def get_cleaning_report(self) -> dict:
        final_shape = self.df.shape
        rows_removed = self.original_shape[0] - final_shape[0]
        cols_removed = self.original_shape[1] - final_shape[1]
        
        return {
            'original_rows': self.original_shape[0],
            'original_columns': self.original_shape[1],
            'final_rows': final_shape[0],
            'final_columns': final_shape[1],
            'rows_removed': rows_removed,
            'columns_removed': cols_removed,
            'missing_values': self.df.isnull().sum().sum()
        }

def clean_dataset(df: pd.DataFrame, operations: List[dict]) -> pd.DataFrame:
    cleaner = DataCleaner(df)
    
    for op in operations:
        method = op.get('method')
        params = op.get('params', {})
        
        if method == 'remove_duplicates':
            cleaner.remove_duplicates(**params)
        elif method == 'handle_missing_values':
            cleaner.handle_missing_values(**params)
        elif method == 'normalize_column':
            cleaner.normalize_column(**params)
        elif method == 'remove_outliers':
            cleaner.remove_outliers(**params)
    
    return cleaner.get_cleaned_data()