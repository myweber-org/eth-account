
import pandas as pd
import numpy as np
from typing import List, Optional

class DataCleaner:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.original_shape = df.shape

    def remove_duplicates(self, subset: Optional[List[str]] = None) -> 'DataCleaner':
        self.df = self.df.drop_duplicates(subset=subset, keep='first')
        return self

    def handle_missing_values(self, strategy: str = 'drop', fill_value: Optional[float] = None) -> 'DataCleaner':
        if strategy == 'drop':
            self.df = self.df.dropna()
        elif strategy == 'fill':
            if fill_value is not None:
                self.df = self.df.fillna(fill_value)
            else:
                numeric_cols = self.df.select_dtypes(include=[np.number]).columns
                self.df[numeric_cols] = self.df[numeric_cols].fillna(self.df[numeric_cols].mean())
        return self

    def normalize_column_names(self) -> 'DataCleaner':
        self.df.columns = [col.strip().lower().replace(' ', '_') for col in self.df.columns]
        return self

    def get_cleaned_data(self) -> pd.DataFrame:
        return self.df

    def get_cleaning_report(self) -> dict:
        return {
            'original_rows': self.original_shape[0],
            'cleaned_rows': self.df.shape[0],
            'original_columns': self.original_shape[1],
            'cleaned_columns': self.df.shape[1],
            'rows_removed': self.original_shape[0] - self.df.shape[0],
            'columns_removed': self.original_shape[1] - self.df.shape[1]
        }

def clean_dataset(df: pd.DataFrame, steps: List[str] = None) -> pd.DataFrame:
    if steps is None:
        steps = ['normalize', 'deduplicate', 'handle_missing']
    
    cleaner = DataCleaner(df)
    
    if 'normalize' in steps:
        cleaner.normalize_column_names()
    
    if 'deduplicate' in steps:
        cleaner.remove_duplicates()
    
    if 'handle_missing' in steps:
        cleaner.handle_missing_values(strategy='fill')
    
    return cleaner.get_cleaned_data()