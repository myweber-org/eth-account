
def deduplicate_list(input_list):
    seen = set()
    result = []
    for item in input_list:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]

def normalize_minmax(data, column):
    min_val = data[column].min()
    max_val = data[column].max()
    data[column + '_normalized'] = (data[column] - min_val) / (max_val - min_val)
    return data

def standardize_zscore(data, column):
    mean_val = data[column].mean()
    std_val = data[column].std()
    data[column + '_standardized'] = (data[column] - mean_val) / std_val
    return data

def clean_dataset(df, numeric_columns, outlier_removal=True, normalization='minmax'):
    cleaned_df = df.copy()
    
    for col in numeric_columns:
        if outlier_removal:
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
        
        if normalization == 'minmax':
            cleaned_df = normalize_minmax(cleaned_df, col)
        elif normalization == 'zscore':
            cleaned_df = standardize_zscore(cleaned_df, col)
    
    return cleaned_df

def generate_sample_data():
    np.random.seed(42)
    data = {
        'feature_a': np.random.normal(100, 15, 200),
        'feature_b': np.random.exponential(50, 200),
        'feature_c': np.random.uniform(0, 1000, 200)
    }
    return pd.DataFrame(data)
import pandas as pd
import numpy as np
from pathlib import Path

class CSVDataCleaner:
    def __init__(self, file_path):
        self.file_path = Path(file_path)
        self.df = None
        self.cleaning_report = {}
        
    def load_data(self):
        try:
            self.df = pd.read_csv(self.file_path)
            self.cleaning_report['original_rows'] = len(self.df)
            self.cleaning_report['original_columns'] = len(self.df.columns)
            return True
        except Exception as e:
            print(f"Error loading file: {e}")
            return False
    
    def remove_duplicates(self):
        if self.df is not None:
            initial_count = len(self.df)
            self.df.drop_duplicates(inplace=True)
            removed = initial_count - len(self.df)
            self.cleaning_report['duplicates_removed'] = removed
            return removed
        return 0
    
    def handle_missing_values(self, strategy='mean', columns=None):
        if self.df is not None:
            if columns is None:
                columns = self.df.select_dtypes(include=[np.number]).columns
            
            missing_before = self.df[columns].isnull().sum().sum()
            
            for col in columns:
                if col in self.df.columns:
                    if strategy == 'mean' and self.df[col].dtype in [np.float64, np.int64]:
                        self.df[col].fillna(self.df[col].mean(), inplace=True)
                    elif strategy == 'median' and self.df[col].dtype in [np.float64, np.int64]:
                        self.df[col].fillna(self.df[col].median(), inplace=True)
                    elif strategy == 'mode':
                        self.df[col].fillna(self.df[col].mode()[0], inplace=True)
                    elif strategy == 'drop':
                        self.df.dropna(subset=[col], inplace=True)
            
            missing_after = self.df[columns].isnull().sum().sum()
            self.cleaning_report['missing_values_handled'] = missing_before - missing_after
            return missing_before - missing_after
        return 0
    
    def normalize_numeric_columns(self, columns=None):
        if self.df is not None:
            if columns is None:
                columns = self.df.select_dtypes(include=[np.number]).columns
            
            normalized_cols = []
            for col in columns:
                if col in self.df.columns:
                    min_val = self.df[col].min()
                    max_val = self.df[col].max()
                    
                    if max_val > min_val:
                        self.df[f'{col}_normalized'] = (self.df[col] - min_val) / (max_val - min_val)
                        normalized_cols.append(col)
            
            self.cleaning_report['normalized_columns'] = normalized_cols
            return normalized_cols
        return []
    
    def save_cleaned_data(self, output_path=None):
        if self.df is not None:
            if output_path is None:
                output_path = self.file_path.parent / f'cleaned_{self.file_path.name}'
            
            self.df.to_csv(output_path, index=False)
            self.cleaning_report['output_file'] = str(output_path)
            return output_path
        return None
    
    def get_summary(self):
        summary = {
            'file': str(self.file_path),
            'final_rows': len(self.df) if self.df is not None else 0,
            'final_columns': len(self.df.columns) if self.df is not None else 0,
            **self.cleaning_report
        }
        return summary

def clean_csv_file(input_file, output_file=None):
    cleaner = CSVDataCleaner(input_file)
    
    if cleaner.load_data():
        cleaner.remove_duplicates()
        cleaner.handle_missing_values(strategy='mean')
        cleaner.normalize_numeric_columns()
        
        saved_path = cleaner.save_cleaned_data(output_file)
        summary = cleaner.get_summary()
        
        print("Data cleaning completed successfully")
        print(f"Cleaned data saved to: {saved_path}")
        
        for key, value in summary.items():
            print(f"{key}: {value}")
        
        return True, summary
    
    return False, None

if __name__ == "__main__":
    sample_data = {
        'id': [1, 2, 3, 4, 5, 5],
        'value': [10.5, None, 15.2, 20.1, None, 10.5],
        'category': ['A', 'B', 'A', None, 'C', 'A']
    }
    
    test_df = pd.DataFrame(sample_data)
    test_df.to_csv('test_data.csv', index=False)
    
    success, report = clean_csv_file('test_data.csv')
    
    import os
    if os.path.exists('test_data.csv'):
        os.remove('test_data.csv')
    
    if os.path.exists('cleaned_test_data.csv'):
        os.remove('cleaned_test_data.csv')