
import pandas as pd
import numpy as np
from typing import Optional, Dict, List

class DataCleaner:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.original_shape = df.shape
        
    def handle_missing_values(self, strategy: str = 'mean', columns: Optional[List[str]] = None) -> 'DataCleaner':
        if columns is None:
            columns = self.df.columns
            
        for col in columns:
            if col in self.df.columns:
                if strategy == 'mean' and self.df[col].dtype in ['int64', 'float64']:
                    self.df[col].fillna(self.df[col].mean(), inplace=True)
                elif strategy == 'median' and self.df[col].dtype in ['int64', 'float64']:
                    self.df[col].fillna(self.df[col].median(), inplace=True)
                elif strategy == 'mode':
                    self.df[col].fillna(self.df[col].mode()[0] if not self.df[col].mode().empty else 0, inplace=True)
                elif strategy == 'drop':
                    self.df.dropna(subset=[col], inplace=True)
                else:
                    self.df[col].fillna(0, inplace=True)
        return self
        
    def convert_types(self, type_map: Dict[str, str]) -> 'DataCleaner':
        for col, dtype in type_map.items():
            if col in self.df.columns:
                try:
                    if dtype == 'datetime':
                        self.df[col] = pd.to_datetime(self.df[col], errors='coerce')
                    elif dtype == 'numeric':
                        self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
                    elif dtype == 'category':
                        self.df[col] = self.df[col].astype('category')
                    else:
                        self.df[col] = self.df[col].astype(dtype)
                except Exception as e:
                    print(f"Error converting column {col} to {dtype}: {e}")
        return self
        
    def remove_duplicates(self, subset: Optional[List[str]] = None, keep: str = 'first') -> 'DataCleaner':
        self.df.drop_duplicates(subset=subset, keep=keep, inplace=True)
        return self
        
    def normalize_columns(self, columns: List[str], method: str = 'minmax') -> 'DataCleaner':
        for col in columns:
            if col in self.df.columns and self.df[col].dtype in ['int64', 'float64']:
                if method == 'minmax':
                    min_val = self.df[col].min()
                    max_val = self.df[col].max()
                    if max_val > min_val:
                        self.df[col] = (self.df[col] - min_val) / (max_val - min_val)
                elif method == 'zscore':
                    mean_val = self.df[col].mean()
                    std_val = self.df[col].std()
                    if std_val > 0:
                        self.df[col] = (self.df[col] - mean_val) / std_val
        return self
        
    def get_cleaned_data(self) -> pd.DataFrame:
        return self.df
        
    def get_cleaning_report(self) -> Dict:
        return {
            'original_shape': self.original_shape,
            'cleaned_shape': self.df.shape,
            'missing_values': self.df.isnull().sum().to_dict(),
            'data_types': self.df.dtypes.astype(str).to_dict()
        }

def clean_csv_file(input_path: str, output_path: str, cleaning_steps: Dict) -> Dict:
    try:
        df = pd.read_csv(input_path)
        cleaner = DataCleaner(df)
        
        if 'missing_values' in cleaning_steps:
            cleaner.handle_missing_values(**cleaning_steps['missing_values'])
            
        if 'type_conversion' in cleaning_steps:
            cleaner.convert_types(cleaning_steps['type_conversion'])
            
        if 'remove_duplicates' in cleaning_steps:
            cleaner.remove_duplicates(**cleaning_steps['remove_duplicates'])
            
        if 'normalize' in cleaning_steps:
            cleaner.normalize_columns(**cleaning_steps['normalize'])
        
        cleaned_df = cleaner.get_cleaned_data()
        cleaned_df.to_csv(output_path, index=False)
        
        report = cleaner.get_cleaning_report()
        report['status'] = 'success'
        return report
        
    except Exception as e:
        return {'status': 'error', 'message': str(e)}
import numpy as np
import pandas as pd

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return filtered_df

def clean_dataset(df, numeric_columns):
    cleaned_df = df.copy()
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
    cleaned_df = cleaned_df.reset_index(drop=True)
    return cleaned_df

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'A': np.random.normal(100, 15, 1000),
        'B': np.random.exponential(50, 1000),
        'C': np.random.uniform(0, 200, 1000)
    })
    numeric_cols = ['A', 'B', 'C']
    cleaned_data = clean_dataset(sample_data, numeric_cols)
    print(f"Original shape: {sample_data.shape}")
    print(f"Cleaned shape: {cleaned_data.shape}")
def remove_duplicates_preserve_order(input_list):
    seen = set()
    result = []
    for item in input_list:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return resultimport csv
import re

def clean_numeric_string(value):
    """Remove non-numeric characters from a string."""
    if not isinstance(value, str):
        return value
    return re.sub(r'[^\d.-]', '', value)

def standardize_phone_number(phone):
    """Standardize phone number format to (XXX) XXX-XXXX."""
    cleaned = clean_numeric_string(str(phone))
    if len(cleaned) == 10:
        return f"({cleaned[:3]}) {cleaned[3:6]}-{cleaned[6:]}"
    return phone

def read_csv_file(filepath):
    """Read CSV file and return data as list of dictionaries."""
    try:
        with open(filepath, 'r', newline='', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            return [row for row in reader]
    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found.")
        return []

def write_csv_file(filepath, data, fieldnames):
    """Write data to CSV file."""
    try:
        with open(filepath, 'w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)
        print(f"Data successfully written to '{filepath}'.")
    except Exception as e:
        print(f"Error writing file: {e}")

def clean_csv_data(input_file, output_file):
    """Clean data in CSV file and save to new file."""
    data = read_csv_file(input_file)
    if not data:
        return
    
    cleaned_data = []
    for row in data:
        cleaned_row = {}
        for key, value in row.items():
            if 'phone' in key.lower():
                cleaned_row[key] = standardize_phone_number(value)
            elif any(num_key in key.lower() for num_key in ['amount', 'price', 'quantity']):
                cleaned_row[key] = clean_numeric_string(value)
            else:
                cleaned_row[key] = value.strip() if isinstance(value, str) else value
        cleaned_data.append(cleaned_row)
    
    if cleaned_data:
        write_csv_file(output_file, cleaned_data, list(cleaned_data[0].keys()))

if __name__ == "__main__":
    input_csv = "raw_data.csv"
    output_csv = "cleaned_data.csv"
    clean_csv_data(input_csv, output_csv)