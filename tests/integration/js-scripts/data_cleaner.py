
import pandas as pd
import numpy as np
from typing import Optional, List, Dict

class DataCleaner:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.original_shape = df.shape
        
    def remove_duplicates(self, subset: Optional[List[str]] = None) -> 'DataCleaner':
        self.df = self.df.drop_duplicates(subset=subset, keep='first')
        return self
        
    def fill_missing_values(self, strategy: str = 'mean', 
                           columns: Optional[List[str]] = None,
                           custom_value: Optional[Dict] = None) -> 'DataCleaner':
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns.tolist()
            
        for col in columns:
            if col not in self.df.columns:
                continue
                
            if custom_value and col in custom_value:
                self.df[col] = self.df[col].fillna(custom_value[col])
            elif strategy == 'mean':
                self.df[col] = self.df[col].fillna(self.df[col].mean())
            elif strategy == 'median':
                self.df[col] = self.df[col].fillna(self.df[col].median())
            elif strategy == 'mode':
                self.df[col] = self.df[col].fillna(self.df[col].mode()[0])
            elif strategy == 'ffill':
                self.df[col] = self.df[col].fillna(method='ffill')
            elif strategy == 'bfill':
                self.df[col] = self.df[col].fillna(method='bfill')
            elif strategy == 'drop':
                self.df = self.df.dropna(subset=[col])
                
        return self
        
    def remove_outliers(self, columns: List[str], 
                       method: str = 'iqr',
                       threshold: float = 1.5) -> 'DataCleaner':
        for col in columns:
            if col not in self.df.columns:
                continue
                
            if method == 'iqr':
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                self.df = self.df[(self.df[col] >= lower_bound) & 
                                 (self.df[col] <= upper_bound)]
            elif method == 'zscore':
                from scipy import stats
                z_scores = np.abs(stats.zscore(self.df[col]))
                self.df = self.df[z_scores < threshold]
                
        return self
        
    def normalize_columns(self, columns: List[str], 
                         method: str = 'minmax') -> 'DataCleaner':
        for col in columns:
            if col not in self.df.columns:
                continue
                
            if method == 'minmax':
                min_val = self.df[col].min()
                max_val = self.df[col].max()
                if max_val != min_val:
                    self.df[col] = (self.df[col] - min_val) / (max_val - min_val)
            elif method == 'standard':
                mean_val = self.df[col].mean()
                std_val = self.df[col].std()
                if std_val != 0:
                    self.df[col] = (self.df[col] - mean_val) / std_val
                    
        return self
        
    def get_cleaning_report(self) -> Dict:
        removed_rows = self.original_shape[0] - self.df.shape[0]
        removed_cols = self.original_shape[1] - self.df.shape[1]
        
        return {
            'original_shape': self.original_shape,
            'cleaned_shape': self.df.shape,
            'removed_rows': removed_rows,
            'removed_cols': removed_cols,
            'missing_values': self.df.isnull().sum().to_dict(),
            'data_types': self.df.dtypes.astype(str).to_dict()
        }
        
    def get_dataframe(self) -> pd.DataFrame:
        return self.df.copy()
        
    def save_cleaned_data(self, filepath: str, format: str = 'csv'):
        if format == 'csv':
            self.df.to_csv(filepath, index=False)
        elif format == 'excel':
            self.df.to_excel(filepath, index=False)
        elif format == 'parquet':
            self.df.to_parquet(filepath, index=False)


def load_and_clean_csv(filepath: str, 
                      cleaning_steps: Optional[List[Dict]] = None) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    cleaner = DataCleaner(df)
    
    if cleaning_steps:
        for step in cleaning_steps:
            method = step.get('method')
            params = step.get('params', {})
            
            if hasattr(cleaner, method):
                getattr(cleaner, method)(**params)
    
    return cleaner.get_dataframe()