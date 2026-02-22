
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
    
    def normalize_column(self, column_name: str, method: str = 'minmax') -> pd.DataFrame:
        if column_name not in self.df.columns:
            raise ValueError(f"Column '{column_name}' not found in DataFrame")
        
        if method == 'minmax':
            col_min = self.df[column_name].min()
            col_max = self.df[column_name].max()
            if col_max != col_min:
                self.df[f"{column_name}_normalized"] = (self.df[column_name] - col_min) / (col_max - col_min)
            else:
                self.df[f"{column_name}_normalized"] = 0
        elif method == 'zscore':
            col_mean = self.df[column_name].mean()
            col_std = self.df[column_name].std()
            if col_std > 0:
                self.df[f"{column_name}_normalized"] = (self.df[column_name] - col_mean) / col_std
            else:
                self.df[f"{column_name}_normalized"] = 0
        else:
            raise ValueError("Method must be 'minmax' or 'zscore'")
        
        return self.df
    
    def handle_missing_values(self, strategy: str = 'mean', columns: Optional[List[str]] = None) -> pd.DataFrame:
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns.tolist()
        
        for col in columns:
            if col not in self.df.columns:
                continue
                
            if strategy == 'mean':
                fill_value = self.df[col].mean()
            elif strategy == 'median':
                fill_value = self.df[col].median()
            elif strategy == 'mode':
                fill_value = self.df[col].mode()[0] if not self.df[col].mode().empty else 0
            elif strategy == 'drop':
                self.df = self.df.dropna(subset=[col])
                continue
            else:
                raise ValueError("Strategy must be 'mean', 'median', 'mode', or 'drop'")
            
            missing_count = self.df[col].isna().sum()
            if missing_count > 0:
                self.df[col] = self.df[col].fillna(fill_value)
                print(f"Filled {missing_count} missing values in column '{col}' with {strategy}: {fill_value}")
        
        return self.df
    
    def get_summary(self) -> dict:
        return {
            'original_shape': self.original_shape,
            'current_shape': self.df.shape,
            'columns': list(self.df.columns),
            'missing_values': self.df.isna().sum().to_dict(),
            'data_types': self.df.dtypes.astype(str).to_dict()
        }
    
    def save_cleaned_data(self, filepath: str):
        self.df.to_csv(filepath, index=False)
        print(f"Cleaned data saved to {filepath}")