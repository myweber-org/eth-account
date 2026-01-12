
import pandas as pd
import numpy as np
from typing import Optional, Dict, List

class DataCleaner:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.original_shape = df.shape
        
    def remove_duplicates(self, subset: Optional[List[str]] = None) -> 'DataCleaner':
        self.df = self.df.drop_duplicates(subset=subset, keep='first')
        return self
        
    def handle_missing_values(self, strategy: str = 'mean', 
                             custom_values: Optional[Dict] = None) -> 'DataCleaner':
        if strategy == 'drop':
            self.df = self.df.dropna()
        elif strategy == 'mean':
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            self.df[numeric_cols] = self.df[numeric_cols].fillna(self.df[numeric_cols].mean())
        elif strategy == 'median':
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            self.df[numeric_cols] = self.df[numeric_cols].fillna(self.df[numeric_cols].median())
        elif strategy == 'custom' and custom_values:
            self.df = self.df.fillna(custom_values)
        return self
        
    def convert_dtypes(self, type_map: Dict[str, str]) -> 'DataCleaner':
        for col, dtype in type_map.items():
            if col in self.df.columns:
                try:
                    self.df[col] = self.df[col].astype(dtype)
                except ValueError as e:
                    print(f"Warning: Could not convert column {col} to {dtype}: {e}")
        return self
        
    def remove_outliers_iqr(self, columns: List[str], multiplier: float = 1.5) -> 'DataCleaner':
        for col in columns:
            if col in self.df.select_dtypes(include=[np.number]).columns:
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - multiplier * IQR
                upper_bound = Q3 + multiplier * IQR
                self.df = self.df[(self.df[col] >= lower_bound) & (self.df[col] <= upper_bound)]
        return self
        
    def normalize_columns(self, columns: List[str], method: str = 'minmax') -> 'DataCleaner':
        for col in columns:
            if col in self.df.select_dtypes(include=[np.number]).columns:
                if method == 'minmax':
                    min_val = self.df[col].min()
                    max_val = self.df[col].max()
                    if max_val != min_val:
                        self.df[col] = (self.df[col] - min_val) / (max_val - min_val)
                elif method == 'zscore':
                    mean_val = self.df[col].mean()
                    std_val = self.df[col].std()
                    if std_val != 0:
                        self.df[col] = (self.df[col] - mean_val) / std_val
        return self
        
    def get_cleaned_data(self) -> pd.DataFrame:
        return self.df
        
    def get_cleaning_report(self) -> Dict:
        return {
            'original_rows': self.original_shape[0],
            'original_columns': self.original_shape[1],
            'cleaned_rows': self.df.shape[0],
            'cleaned_columns': self.df.shape[1],
            'rows_removed': self.original_shape[0] - self.df.shape[0],
            'columns_removed': self.original_shape[1] - self.df.shape[1],
            'missing_values': self.df.isnull().sum().sum(),
            'duplicates_removed': self.original_shape[0] - self.df.shape[0]
        }

def clean_csv_file(input_path: str, output_path: str, 
                  cleaning_steps: Optional[Dict] = None) -> Dict:
    try:
        df = pd.read_csv(input_path)
        cleaner = DataCleaner(df)
        
        if cleaning_steps:
            if cleaning_steps.get('remove_duplicates'):
                cleaner.remove_duplicates(cleaning_steps.get('duplicate_subset'))
            if cleaning_steps.get('handle_missing'):
                cleaner.handle_missing_values(
                    strategy=cleaning_steps.get('missing_strategy', 'mean'),
                    custom_values=cleaning_steps.get('custom_fill_values')
                )
            if cleaning_steps.get('convert_dtypes'):
                cleaner.convert_dtypes(cleaning_steps.get('type_map', {}))
            if cleaning_steps.get('remove_outliers'):
                cleaner.remove_outliers_iqr(
                    columns=cleaning_steps.get('outlier_columns', []),
                    multiplier=cleaning_steps.get('outlier_multiplier', 1.5)
                )
            if cleaning_steps.get('normalize'):
                cleaner.normalize_columns(
                    columns=cleaning_steps.get('normalize_columns', []),
                    method=cleaning_steps.get('normalize_method', 'minmax')
                )
        
        cleaned_df = cleaner.get_cleaned_data()
        cleaned_df.to_csv(output_path, index=False)
        
        report = cleaner.get_cleaning_report()
        report['status'] = 'success'
        return report
        
    except Exception as e:
        return {
            'status': 'error',
            'error_message': str(e)
        }