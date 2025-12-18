import pandas as pd
import numpy as np

def clean_csv_data(input_file, output_file, missing_strategy='mean'):
    """
    Clean CSV data by handling missing values and removing duplicates.
    
    Args:
        input_file (str): Path to input CSV file
        output_file (str): Path to save cleaned CSV file
        missing_strategy (str): Strategy for handling missing values
                               Options: 'mean', 'median', 'drop', 'zero'
    """
    try:
        df = pd.read_csv(input_file)
        
        print(f"Original data shape: {df.shape}")
        print(f"Missing values per column:\n{df.isnull().sum()}")
        
        df_cleaned = df.copy()
        
        df_cleaned = df_cleaned.drop_duplicates()
        print(f"After removing duplicates: {df_cleaned.shape}")
        
        numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
        
        if missing_strategy == 'mean':
            for col in numeric_cols:
                if df_cleaned[col].isnull().any():
                    df_cleaned[col].fillna(df_cleaned[col].mean(), inplace=True)
        elif missing_strategy == 'median':
            for col in numeric_cols:
                if df_cleaned[col].isnull().any():
                    df_cleaned[col].fillna(df_cleaned[col].median(), inplace=True)
        elif missing_strategy == 'zero':
            df_cleaned.fillna(0, inplace=True)
        elif missing_strategy == 'drop':
            df_cleaned.dropna(inplace=True)
        
        categorical_cols = df_cleaned.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df_cleaned[col].isnull().any():
                df_cleaned[col].fillna('Unknown', inplace=True)
        
        print(f"Final cleaned data shape: {df_cleaned.shape}")
        print(f"Missing values after cleaning:\n{df_cleaned.isnull().sum().sum()}")
        
        df_cleaned.to_csv(output_file, index=False)
        print(f"Cleaned data saved to: {output_file}")
        
        return df_cleaned
        
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
        return None
    except Exception as e:
        print(f"Error during data cleaning: {str(e)}")
        return None

def validate_data(df, required_columns=None):
    """
    Validate cleaned data for basic quality checks.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        required_columns (list): List of required column names
    
    Returns:
        dict: Validation results
    """
    validation_results = {
        'is_valid': True,
        'issues': []
    }
    
    if df is None or df.empty:
        validation_results['is_valid'] = False
        validation_results['issues'].append('DataFrame is empty or None')
        return validation_results
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            validation_results['is_valid'] = False
            validation_results['issues'].append(f'Missing required columns: {missing_cols}')
    
    if df.isnull().sum().sum() > 0:
        validation_results['is_valid'] = False
        validation_results['issues'].append('Data contains missing values')
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].min() < 0 and col not in ['temperature', 'balance']:
            validation_results['issues'].append(f'Column {col} contains negative values')
    
    return validation_results

if __name__ == "__main__":
    input_csv = "raw_data.csv"
    output_csv = "cleaned_data.csv"
    
    cleaned_data = clean_csv_data(input_csv, output_csv, missing_strategy='mean')
    
    if cleaned_data is not None:
        validation = validate_data(cleaned_data)
        print(f"Data validation passed: {validation['is_valid']}")
        if not validation['is_valid']:
            print(f"Validation issues: {validation['issues']}")
import pandas as pd
import numpy as np
from pathlib import Path

class DataCleaner:
    def __init__(self, file_path):
        self.file_path = Path(file_path)
        self.df = None
        
    def load_data(self):
        if self.file_path.exists():
            self.df = pd.read_csv(self.file_path)
            return True
        return False
    
    def remove_duplicates(self):
        if self.df is not None:
            initial_rows = len(self.df)
            self.df = self.df.drop_duplicates()
            removed = initial_rows - len(self.df)
            return removed
        return 0
    
    def fill_missing_values(self, strategy='mean', fill_value=None):
        if self.df is not None:
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            
            if strategy == 'mean':
                for col in numeric_cols:
                    self.df[col] = self.df[col].fillna(self.df[col].mean())
            elif strategy == 'median':
                for col in numeric_cols:
                    self.df[col] = self.df[col].fillna(self.df[col].median())
            elif strategy == 'custom' and fill_value is not None:
                self.df = self.df.fillna(fill_value)
            
            return len(numeric_cols)
        return 0
    
    def remove_outliers(self, threshold=3):
        if self.df is not None:
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            initial_rows = len(self.df)
            
            for col in numeric_cols:
                z_scores = np.abs((self.df[col] - self.df[col].mean()) / self.df[col].std())
                self.df = self.df[z_scores < threshold]
            
            removed = initial_rows - len(self.df)
            return removed
        return 0
    
    def save_cleaned_data(self, output_path=None):
        if self.df is not None:
            if output_path is None:
                output_path = self.file_path.parent / f"cleaned_{self.file_path.name}"
            
            self.df.to_csv(output_path, index=False)
            return str(output_path)
        return None
    
    def get_summary(self):
        if self.df is not None:
            summary = {
                'rows': len(self.df),
                'columns': len(self.df.columns),
                'missing_values': self.df.isnull().sum().sum(),
                'data_types': self.df.dtypes.to_dict()
            }
            return summary
        return None

def process_csv_file(input_file, output_dir='cleaned_data'):
    cleaner = DataCleaner(input_file)
    
    if not cleaner.load_data():
        print(f"Error: File {input_file} not found")
        return None
    
    print(f"Loaded data: {len(cleaner.df)} rows, {len(cleaner.df.columns)} columns")
    
    duplicates_removed = cleaner.remove_duplicates()
    print(f"Removed {duplicates_removed} duplicate rows")
    
    cols_filled = cleaner.fill_missing_values(strategy='mean')
    print(f"Filled missing values in {cols_filled} numeric columns")
    
    outliers_removed = cleaner.remove_outliers()
    print(f"Removed {outliers_removed} outlier rows")
    
    output_path = cleaner.save_cleaned_data()
    print(f"Saved cleaned data to: {output_path}")
    
    summary = cleaner.get_summary()
    print(f"Final dataset: {summary['rows']} rows, {summary['columns']} columns")
    
    return output_path