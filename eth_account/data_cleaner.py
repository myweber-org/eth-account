
import pandas as pd
import numpy as np
from typing import Optional

class DataCleaner:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.original_shape = df.shape
        
    def remove_duplicates(self) -> 'DataCleaner':
        self.df = self.df.drop_duplicates()
        return self
        
    def fill_missing_numeric(self, strategy: str = 'mean', custom_value: Optional[float] = None) -> 'DataCleaner':
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        if strategy == 'mean':
            for col in numeric_cols:
                self.df[col] = self.df[col].fillna(self.df[col].mean())
        elif strategy == 'median':
            for col in numeric_cols:
                self.df[col] = self.df[col].fillna(self.df[col].median())
        elif strategy == 'custom' and custom_value is not None:
            for col in numeric_cols:
                self.df[col] = self.df[col].fillna(custom_value)
                
        return self
        
    def fill_missing_categorical(self, strategy: str = 'mode', custom_value: Optional[str] = None) -> 'DataCleaner':
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns
        
        if strategy == 'mode':
            for col in categorical_cols:
                mode_value = self.df[col].mode()
                if not mode_value.empty:
                    self.df[col] = self.df[col].fillna(mode_value.iloc[0])
        elif strategy == 'custom' and custom_value is not None:
            for col in categorical_cols:
                self.df[col] = self.df[col].fillna(custom_value)
                
        return self
        
    def remove_outliers_iqr(self, columns: list, multiplier: float = 1.5) -> 'DataCleaner':
        for col in columns:
            if col in self.df.columns and pd.api.types.is_numeric_dtype(self.df[col]):
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - multiplier * IQR
                upper_bound = Q3 + multiplier * IQR
                
                self.df = self.df[(self.df[col] >= lower_bound) & (self.df[col] <= upper_bound)]
                
        return self
        
    def get_cleaned_data(self) -> pd.DataFrame:
        return self.df
        
    def get_cleaning_report(self) -> dict:
        removed_rows = self.original_shape[0] - self.df.shape[0]
        removed_cols = self.original_shape[1] - self.df.shape[1]
        
        return {
            'original_shape': self.original_shape,
            'cleaned_shape': self.df.shape,
            'removed_rows': removed_rows,
            'removed_columns': removed_cols,
            'missing_values': self.df.isnull().sum().sum()
        }

def load_and_clean_csv(filepath: str, **cleaner_kwargs) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    cleaner = DataCleaner(df)
    
    if 'remove_dups' in cleaner_kwargs and cleaner_kwargs['remove_dups']:
        cleaner.remove_duplicates()
        
    if 'numeric_strategy' in cleaner_kwargs:
        cleaner.fill_missing_numeric(strategy=cleaner_kwargs['numeric_strategy'])
        
    if 'categorical_strategy' in cleaner_kwargs:
        cleaner.fill_missing_categorical(strategy=cleaner_kwargs['categorical_strategy'])
        
    return cleaner.get_cleaned_data()