
import pandas as pd
import numpy as np
from pathlib import Path

class DataCleaner:
    def __init__(self, file_path):
        self.file_path = Path(file_path)
        self.df = None
        
    def load_data(self):
        try:
            self.df = pd.read_csv(self.file_path)
            print(f"Loaded data with shape: {self.df.shape}")
            return True
        except Exception as e:
            print(f"Error loading file: {e}")
            return False
    
    def handle_missing_values(self, strategy='mean', columns=None):
        if self.df is None:
            print("No data loaded. Call load_data() first.")
            return
        
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        for col in columns:
            if col in self.df.columns and self.df[col].isnull().any():
                if strategy == 'mean':
                    fill_value = self.df[col].mean()
                elif strategy == 'median':
                    fill_value = self.df[col].median()
                elif strategy == 'mode':
                    fill_value = self.df[col].mode()[0]
                elif strategy == 'zero':
                    fill_value = 0
                else:
                    continue
                
                self.df[col].fillna(fill_value, inplace=True)
                print(f"Filled missing values in '{col}' using {strategy}")
    
    def remove_duplicates(self, subset=None, keep='first'):
        if self.df is None:
            print("No data loaded. Call load_data() first.")
            return
        
        initial_count = len(self.df)
        self.df.drop_duplicates(subset=subset, keep=keep, inplace=True)
        removed_count = initial_count - len(self.df)
        print(f"Removed {removed_count} duplicate rows")
    
    def normalize_columns(self, columns=None):
        if self.df is None:
            print("No data loaded. Call load_data() first.")
            return
        
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        for col in columns:
            if col in self.df.columns:
                col_min = self.df[col].min()
                col_max = self.df[col].max()
                if col_max > col_min:
                    self.df[col] = (self.df[col] - col_min) / (col_max - col_min)
                    print(f"Normalized column '{col}' to range [0, 1]")
    
    def save_cleaned_data(self, output_path=None):
        if self.df is None:
            print("No data to save. Call load_data() first.")
            return
        
        if output_path is None:
            output_path = self.file_path.parent / f"cleaned_{self.file_path.name}"
        
        self.df.to_csv(output_path, index=False)
        print(f"Saved cleaned data to: {output_path}")
        return output_path
    
    def get_summary(self):
        if self.df is None:
            return "No data loaded"
        
        summary = {
            'shape': self.df.shape,
            'missing_values': self.df.isnull().sum().sum(),
            'duplicates': self.df.duplicated().sum(),
            'data_types': self.df.dtypes.value_counts().to_dict(),
            'numeric_columns': list(self.df.select_dtypes(include=[np.number]).columns),
            'categorical_columns': list(self.df.select_dtypes(include=['object']).columns)
        }
        return summary

def clean_csv_file(input_file, output_file=None):
    cleaner = DataCleaner(input_file)
    
    if cleaner.load_data():
        cleaner.handle_missing_values(strategy='mean')
        cleaner.remove_duplicates()
        cleaner.normalize_columns()
        
        saved_path = cleaner.save_cleaned_data(output_file)
        summary = cleaner.get_summary()
        
        print("\nData cleaning completed successfully!")
        print(f"Original shape: {summary['shape']}")
        print(f"Missing values handled: {summary['missing_values']}")
        print(f"Duplicates removed: {summary['duplicates']}")
        
        return saved_path
    return None