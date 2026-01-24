
import pandas as pd
import numpy as np
from pathlib import Path

class CSVDataCleaner:
    def __init__(self, file_path):
        self.file_path = Path(file_path)
        self.df = None
        
    def load_data(self):
        try:
            self.df = pd.read_csv(self.file_path)
            print(f"Loaded {len(self.df)} rows from {self.file_path.name}")
            return True
        except FileNotFoundError:
            print(f"Error: File {self.file_path} not found")
            return False
        except Exception as e:
            print(f"Error loading file: {e}")
            return False
    
    def show_missing_summary(self):
        if self.df is None:
            print("No data loaded")
            return
        
        missing_counts = self.df.isnull().sum()
        missing_percent = (missing_counts / len(self.df)) * 100
        
        print("\nMissing Value Summary:")
        print("-" * 40)
        for col in self.df.columns:
            if missing_counts[col] > 0:
                print(f"{col}: {missing_counts[col]} missing ({missing_percent[col]:.1f}%)")
    
    def fill_missing_numeric(self, strategy='mean'):
        if self.df is None:
            print("No data loaded")
            return
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if self.df[col].isnull().any():
                if strategy == 'mean':
                    fill_value = self.df[col].mean()
                elif strategy == 'median':
                    fill_value = self.df[col].median()
                elif strategy == 'zero':
                    fill_value = 0
                else:
                    continue
                
                self.df[col].fillna(fill_value, inplace=True)
                print(f"Filled missing values in {col} with {strategy} value: {fill_value:.2f}")
    
    def drop_columns_with_high_missing(self, threshold=50):
        if self.df is None:
            print("No data loaded")
            return
        
        missing_percent = (self.df.isnull().sum() / len(self.df)) * 100
        cols_to_drop = missing_percent[missing_percent > threshold].index.tolist()
        
        if cols_to_drop:
            self.df.drop(columns=cols_to_drop, inplace=True)
            print(f"Dropped columns with >{threshold}% missing values: {cols_to_drop}")
        else:
            print(f"No columns with >{threshold}% missing values found")
    
    def save_cleaned_data(self, output_path=None):
        if self.df is None:
            print("No data to save")
            return
        
        if output_path is None:
            output_path = self.file_path.parent / f"cleaned_{self.file_path.name}"
        
        self.df.to_csv(output_path, index=False)
        print(f"Saved cleaned data to {output_path}")
        return output_path
    
    def get_summary(self):
        if self.df is None:
            return {}
        
        return {
            'original_rows': len(self.df),
            'columns': list(self.df.columns),
            'numeric_columns': list(self.df.select_dtypes(include=[np.number]).columns),
            'categorical_columns': list(self.df.select_dtypes(include=['object']).columns),
            'missing_values': int(self.df.isnull().sum().sum())
        }

def process_csv_file(input_file, output_dir='cleaned_data'):
    cleaner = CSVDataCleaner(input_file)
    
    if not cleaner.load_data():
        return
    
    print("\nInitial data summary:")
    summary = cleaner.get_summary()
    for key, value in summary.items():
        print(f"{key}: {value}")
    
    cleaner.show_missing_summary()
    cleaner.drop_columns_with_high_missing(threshold=60)
    cleaner.fill_missing_numeric(strategy='median')
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    output_path = cleaner.save_cleaned_data(output_dir / f"cleaned_{Path(input_file).name}")
    
    print("\nCleaning completed successfully")
    return output_path