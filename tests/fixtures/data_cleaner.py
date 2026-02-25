
import pandas as pd
import numpy as np
from typing import List, Optional

class DataCleaner:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.original_shape = df.shape
    
    def remove_duplicates(self, subset: Optional[List[str]] = None) -> pd.DataFrame:
        initial_count = len(self.df)
        self.df = self.df.drop_duplicates(subset=subset)
        removed = initial_count - len(self.df)
        print(f"Removed {removed} duplicate rows")
        return self.df
    
    def handle_missing_values(self, strategy: str = 'drop', fill_value: Optional[float] = None) -> pd.DataFrame:
        if strategy == 'drop':
            self.df = self.df.dropna()
        elif strategy == 'fill':
            if fill_value is not None:
                self.df = self.df.fillna(fill_value)
            else:
                self.df = self.df.fillna(self.df.mean())
        print(f"Missing values handled using '{strategy}' strategy")
        return self.df
    
    def normalize_column(self, column: str) -> pd.DataFrame:
        if column in self.df.columns:
            col_min = self.df[column].min()
            col_max = self.df[column].max()
            if col_max != col_min:
                self.df[column] = (self.df[column] - col_min) / (col_max - col_min)
        return self.df
    
    def get_summary(self) -> dict:
        return {
            'original_shape': self.original_shape,
            'current_shape': self.df.shape,
            'missing_values': self.df.isnull().sum().to_dict(),
            'column_types': self.df.dtypes.to_dict()
        }
    
    def save_cleaned_data(self, filepath: str):
        self.df.to_csv(filepath, index=False)
        print(f"Cleaned data saved to {filepath}")

def example_usage():
    data = {
        'id': [1, 2, 2, 3, 4],
        'value': [10.5, None, 15.2, 20.1, None],
        'category': ['A', 'B', 'B', 'A', 'C']
    }
    df = pd.DataFrame(data)
    
    cleaner = DataCleaner(df)
    cleaner.remove_duplicates(subset=['id'])
    cleaner.handle_missing_values(strategy='fill', fill_value=0)
    cleaner.normalize_column('value')
    
    print(cleaner.get_summary())
    return cleaner.df

if __name__ == "__main__":
    cleaned_df = example_usage()
    print(cleaned_df)