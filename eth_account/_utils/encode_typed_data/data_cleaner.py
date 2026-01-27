
import pandas as pd
import numpy as np
from typing import List, Optional

class DataCleaner:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.original_shape = df.shape
        
    def remove_duplicates(self, subset: Optional[List[str]] = None) -> pd.DataFrame:
        initial_count = len(self.df)
        self.df = self.df.drop_duplicates(subset=subset, keep='first')
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
                self.df = self.df.fillna(self.df.mean(numeric_only=True))
        else:
            raise ValueError("Strategy must be 'drop' or 'fill'")
        
        print(f"Missing values handled using '{strategy}' strategy")
        return self.df
    
    def normalize_numeric_columns(self) -> pd.DataFrame:
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if self.df[col].std() != 0:
                self.df[col] = (self.df[col] - self.df[col].mean()) / self.df[col].std()
        print(f"Normalized {len(numeric_cols)} numeric columns")
        return self.df
    
    def get_summary(self) -> dict:
        return {
            'original_rows': self.original_shape[0],
            'current_rows': len(self.df),
            'original_columns': self.original_shape[1],
            'current_columns': len(self.df.columns),
            'missing_values': self.df.isnull().sum().sum(),
            'duplicates_removed': self.original_shape[0] - len(self.df)
        }
    
    def save_cleaned_data(self, filepath: str):
        self.df.to_csv(filepath, index=False)
        print(f"Cleaned data saved to {filepath}")

def example_usage():
    data = {
        'id': [1, 2, 2, 3, 4],
        'value': [10.5, None, 20.3, 15.7, None],
        'category': ['A', 'B', 'B', 'A', 'C']
    }
    
    df = pd.DataFrame(data)
    cleaner = DataCleaner(df)
    
    cleaner.remove_duplicates(subset=['id'])
    cleaner.handle_missing_values(strategy='fill', fill_value=0)
    cleaner.normalize_numeric_columns()
    
    summary = cleaner.get_summary()
    print("Cleaning summary:", summary)
    
    return cleaner.df

if __name__ == "__main__":
    cleaned_df = example_usage()
    print("Cleaned DataFrame:")
    print(cleaned_df)