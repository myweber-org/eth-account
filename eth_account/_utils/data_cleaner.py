
import pandas as pd
import numpy as np
from scipy import stats

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def normalize_minmax(df, column):
    min_val = df[column].min()
    max_val = df[column].max()
    df[column + '_normalized'] = (df[column] - min_val) / (max_val - min_val)
    return df

def clean_dataset(file_path, output_path):
    df = pd.read_csv(file_path)
    
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_columns:
        df = remove_outliers_iqr(df, col)
    
    for col in numeric_columns:
        df = normalize_minmax(df, col)
    
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to {output_path}")
    return df

if __name__ == "__main__":
    input_file = "raw_data.csv"
    output_file = "cleaned_data.csv"
    cleaned_df = clean_dataset(input_file, output_file)import pandas as pd
import numpy as np
from typing import Optional, Dict, List

class DataCleaner:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.original_shape = df.shape
        
    def remove_duplicates(self, subset: Optional[List[str]] = None) -> 'DataCleaner':
        self.df = self.df.drop_duplicates(subset=subset)
        return self
        
    def handle_missing_values(self, 
                             strategy: str = 'mean',
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
        
    def convert_dtypes(self, 
                      date_columns: Optional[List[str]] = None,
                      categorical_columns: Optional[List[str]] = None) -> 'DataCleaner':
        if date_columns:
            for col in date_columns:
                if col in self.df.columns:
                    self.df[col] = pd.to_datetime(self.df[col], errors='coerce')
                    
        if categorical_columns:
            for col in categorical_columns:
                if col in self.df.columns:
                    self.df[col] = self.df[col].astype('category')
                    
        return self
        
    def remove_outliers(self, 
                       columns: List[str],
                       method: str = 'iqr',
                       threshold: float = 1.5) -> 'DataCleaner':
        for col in columns:
            if col not in self.df.columns or self.df[col].dtype not in [np.number]:
                continue
                
            if method == 'iqr':
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                self.df = self.df[(self.df[col] >= lower_bound) & (self.df[col] <= upper_bound)]
                
        return self
        
    def standardize_columns(self, 
                          columns: List[str],
                          method: str = 'zscore') -> 'DataCleaner':
        for col in columns:
            if col not in self.df.columns or self.df[col].dtype not in [np.number]:
                continue
                
            if method == 'zscore':
                self.df[col] = (self.df[col] - self.df[col].mean()) / self.df[col].std()
            elif method == 'minmax':
                self.df[col] = (self.df[col] - self.df[col].min()) / (self.df[col].max() - self.df[col].min())
                
        return self
        
    def get_cleaned_data(self) -> pd.DataFrame:
        return self.df
        
    def get_cleaning_report(self) -> Dict:
        return {
            'original_shape': self.original_shape,
            'cleaned_shape': self.df.shape,
            'rows_removed': self.original_shape[0] - self.df.shape[0],
            'columns_removed': self.original_shape[1] - self.df.shape[1],
            'missing_values_remaining': self.df.isnull().sum().sum(),
            'duplicates_remaining': self.df.duplicated().sum()
        }

def clean_csv_file(input_path: str,
                  output_path: str,
                  missing_strategy: str = 'mean',
                  date_columns: Optional[List[str]] = None,
                  outlier_columns: Optional[List[str]] = None) -> Dict:
    df = pd.read_csv(input_path)
    cleaner = DataCleaner(df)
    
    cleaner.remove_duplicates()
    cleaner.handle_missing_values(strategy=missing_strategy)
    
    if date_columns:
        cleaner.convert_dtypes(date_columns=date_columns)
        
    if outlier_columns:
        cleaner.remove_outliers(columns=outlier_columns)
        
    cleaned_df = cleaner.get_cleaned_data()
    cleaned_df.to_csv(output_path, index=False)
    
    return cleaner.get_cleaning_report()