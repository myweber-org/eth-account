
import pandas as pd
import numpy as np
from typing import List, Optional

class DataCleaner:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.original_shape = df.shape
        
    def remove_duplicates(self, subset: Optional[List[str]] = None) -> 'DataCleaner':
        self.df = self.df.drop_duplicates(subset=subset)
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
        
    def remove_outliers(self, column: str, threshold: float = 3.0) -> 'DataCleaner':
        if column in self.df.columns:
            z_scores = np.abs((self.df[column] - self.df[column].mean()) / self.df[column].std())
            self.df = self.df[z_scores < threshold]
        return self
        
    def get_cleaned_data(self) -> pd.DataFrame:
        return self.df
        
    def get_cleaning_report(self) -> dict:
        return {
            'original_rows': self.original_shape[0],
            'original_columns': self.original_shape[1],
            'cleaned_rows': self.df.shape[0],
            'cleaned_columns': self.df.shape[1],
            'rows_removed': self.original_shape[0] - self.df.shape[0],
            'columns_removed': self.original_shape[1] - self.df.shape[1]
        }

def clean_dataset(data: pd.DataFrame, 
                  remove_dups: bool = True,
                  handle_nulls: bool = True,
                  outlier_columns: Optional[List[str]] = None) -> pd.DataFrame:
    
    cleaner = DataCleaner(data)
    
    if remove_dups:
        cleaner.remove_duplicates()
    
    if handle_nulls:
        cleaner.handle_missing_values(strategy='fill')
    
    if outlier_columns:
        for col in outlier_columns:
            if col in data.columns:
                cleaner.remove_outliers(col)
    
    report = cleaner.get_cleaning_report()
    print(f"Data cleaning completed. Removed {report['rows_removed']} rows.")
    
    return cleaner.get_cleaned_data()