
import pandas as pd
import numpy as np
from typing import List, Optional

class DataCleaner:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.original_shape = df.shape
        
    def remove_duplicates(self, subset: Optional[List[str]] = None) -> pd.DataFrame:
        """Remove duplicate rows from the dataframe."""
        initial_count = len(self.df)
        self.df = self.df.drop_duplicates(subset=subset, keep='first')
        removed = initial_count - len(self.df)
        print(f"Removed {removed} duplicate rows")
        return self.df
    
    def normalize_text_columns(self, columns: List[str]) -> pd.DataFrame:
        """Normalize text columns to lowercase and strip whitespace."""
        for col in columns:
            if col in self.df.columns:
                self.df[col] = self.df[col].astype(str).str.lower().str.strip()
        print(f"Normalized text in columns: {columns}")
        return self.df
    
    def fill_missing_values(self, strategy: str = 'mean', custom_value=None) -> pd.DataFrame:
        """Fill missing values using specified strategy."""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        if strategy == 'mean' and len(numeric_cols) > 0:
            self.df[numeric_cols] = self.df[numeric_cols].fillna(self.df[numeric_cols].mean())
        elif strategy == 'median' and len(numeric_cols) > 0:
            self.df[numeric_cols] = self.df[numeric_cols].fillna(self.df[numeric_cols].median())
        elif strategy == 'custom' and custom_value is not None:
            self.df = self.df.fillna(custom_value)
        
        print(f"Filled missing values using {strategy} strategy")
        return self.df
    
    def remove_outliers_iqr(self, columns: List[str], multiplier: float = 1.5) -> pd.DataFrame:
        """Remove outliers using IQR method."""
        initial_count = len(self.df)
        
        for col in columns:
            if col in self.df.columns:
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - multiplier * IQR
                upper_bound = Q3 + multiplier * IQR
                
                self.df = self.df[(self.df[col] >= lower_bound) & (self.df[col] <= upper_bound)]
        
        removed = initial_count - len(self.df)
        print(f"Removed {removed} outliers from columns: {columns}")
        return self.df
    
    def get_cleaning_report(self) -> dict:
        """Generate a report of cleaning operations."""
        final_shape = self.df.shape
        return {
            'original_rows': self.original_shape[0],
            'original_columns': self.original_shape[1],
            'final_rows': final_shape[0],
            'final_columns': final_shape[1],
            'rows_removed': self.original_shape[0] - final_shape[0],
            'columns_removed': self.original_shape[1] - final_shape[1]
        }
    
    def save_cleaned_data(self, filepath: str) -> None:
        """Save cleaned dataframe to file."""
        self.df.to_csv(filepath, index=False)
        print(f"Cleaned data saved to {filepath}")

def clean_dataset(input_file: str, output_file: str) -> pd.DataFrame:
    """Convenience function for basic dataset cleaning."""
    df = pd.read_csv(input_file)
    cleaner = DataCleaner(df)
    
    cleaner.remove_duplicates()
    cleaner.fill_missing_values(strategy='mean')
    cleaner.normalize_text_columns(df.select_dtypes(include=['object']).columns.tolist())
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        cleaner.remove_outliers_iqr(numeric_cols[:3])
    
    report = cleaner.get_cleaning_report()
    print("Cleaning Report:", report)
    
    cleaner.save_cleaned_data(output_file)
    return cleaner.dfimport pandas as pd
import numpy as np

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

def clean_dataset(input_file, output_file):
    df = pd.read_csv(input_file)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        df = remove_outliers_iqr(df, col)
    
    for col in numeric_cols:
        df = normalize_minmax(df, col)
    
    df.to_csv(output_file, index=False)
    print(f"Cleaned data saved to {output_file}")
    print(f"Original shape: {pd.read_csv(input_file).shape}")
    print(f"Cleaned shape: {df.shape}")

if __name__ == "__main__":
    clean_dataset('raw_data.csv', 'cleaned_data.csv')