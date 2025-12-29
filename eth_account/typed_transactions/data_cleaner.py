import pandas as pd
import numpy as np

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.original_shape = df.shape

    def handle_missing_values(self, strategy='drop', fill_value=None):
        if strategy == 'drop':
            self.df = self.df.dropna()
        elif strategy == 'fill':
            if fill_value is not None:
                self.df = self.df.fillna(fill_value)
            else:
                for col in self.df.select_dtypes(include=[np.number]).columns:
                    self.df[col] = self.df[col].fillna(self.df[col].median())
                for col in self.df.select_dtypes(exclude=[np.number]).columns:
                    self.df[col] = self.df[col].fillna(self.df[col].mode()[0] if not self.df[col].mode().empty else 'Unknown')
        return self

    def remove_duplicates(self, subset=None, keep='first'):
        self.df = self.df.drop_duplicates(subset=subset, keep=keep)
        return self

    def normalize_text_columns(self, columns):
        for col in columns:
            if col in self.df.columns:
                self.df[col] = self.df[col].astype(str).str.lower().str.strip()
        return self

    def get_cleaned_data(self):
        return self.df

    def get_cleaning_report(self):
        final_shape = self.df.shape
        rows_removed = self.original_shape[0] - final_shape[0]
        cols_removed = self.original_shape[1] - final_shape[1]
        
        report = {
            'original_rows': self.original_shape[0],
            'original_columns': self.original_shape[1],
            'final_rows': final_shape[0],
            'final_columns': final_shape[1],
            'rows_removed': rows_removed,
            'columns_removed': cols_removed,
            'missing_values': self.df.isnull().sum().sum(),
            'duplicates': self.df.duplicated().sum()
        }
        return report

def clean_dataset(df, missing_strategy='fill', remove_duplicates=True):
    cleaner = DataCleaner(df)
    cleaner.handle_missing_values(strategy=missing_strategy)
    
    if remove_duplicates:
        cleaner.remove_duplicates()
    
    return cleaner.get_cleaned_data(), cleaner.get_cleaning_report()