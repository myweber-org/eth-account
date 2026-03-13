
import pandas as pd
import numpy as np

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.numeric_columns = self.df.select_dtypes(include=[np.number]).columns
        self.categorical_columns = self.df.select_dtypes(include=['object']).columns

    def handle_missing_values(self, strategy='mean', fill_value=None):
        if strategy == 'mean':
            self.df[self.numeric_columns] = self.df[self.numeric_columns].fillna(self.df[self.numeric_columns].mean())
        elif strategy == 'median':
            self.df[self.numeric_columns] = self.df[self.numeric_columns].fillna(self.df[self.numeric_columns].median())
        elif strategy == 'mode':
            self.df[self.numeric_columns] = self.df[self.numeric_columns].fillna(self.df[self.numeric_columns].mode().iloc[0])
        elif strategy == 'constant' and fill_value is not None:
            self.df[self.numeric_columns] = self.df[self.numeric_columns].fillna(fill_value)
        
        self.df[self.categorical_columns] = self.df[self.categorical_columns].fillna('Unknown')
        return self

    def remove_outliers_iqr(self, columns=None, multiplier=1.5):
        if columns is None:
            columns = self.numeric_columns
        
        for col in columns:
            if col in self.numeric_columns:
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - multiplier * IQR
                upper_bound = Q3 + multiplier * IQR
                
                self.df = self.df[(self.df[col] >= lower_bound) & (self.df[col] <= upper_bound)]
        
        return self

    def standardize_data(self, columns=None):
        if columns is None:
            columns = self.numeric_columns
        
        for col in columns:
            if col in self.numeric_columns:
                mean = self.df[col].mean()
                std = self.df[col].std()
                if std > 0:
                    self.df[col] = (self.df[col] - mean) / std
        
        return self

    def normalize_data(self, columns=None):
        if columns is None:
            columns = self.numeric_columns
        
        for col in columns:
            if col in self.numeric_columns:
                min_val = self.df[col].min()
                max_val = self.df[col].max()
                if max_val > min_val:
                    self.df[col] = (self.df[col] - min_val) / (max_val - min_val)
        
        return self

    def get_cleaned_data(self):
        return self.df

def load_and_clean_data(filepath):
    df = pd.read_csv(filepath)
    cleaner = DataCleaner(df)
    
    cleaner.handle_missing_values(strategy='mean')
    cleaner.remove_outliers_iqr()
    cleaner.standardize_data()
    
    return cleaner.get_cleaned_data()
import pandas as pd
import numpy as np
from scipy import stats

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.original_shape = df.shape
        
    def remove_duplicates(self):
        self.df = self.df.drop_duplicates()
        return self
        
    def handle_missing_values(self, strategy='mean', fill_value=None):
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        if strategy == 'mean':
            self.df[numeric_cols] = self.df[numeric_cols].fillna(self.df[numeric_cols].mean())
        elif strategy == 'median':
            self.df[numeric_cols] = self.df[numeric_cols].fillna(self.df[numeric_cols].median())
        elif strategy == 'mode':
            self.df[numeric_cols] = self.df[numeric_cols].fillna(self.df[numeric_cols].mode().iloc[0])
        elif strategy == 'constant' and fill_value is not None:
            self.df[numeric_cols] = self.df[numeric_cols].fillna(fill_value)
        elif strategy == 'drop':
            self.df = self.df.dropna(subset=numeric_cols)
            
        return self
        
    def detect_outliers_zscore(self, threshold=3):
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        z_scores = np.abs(stats.zscore(self.df[numeric_cols]))
        outlier_mask = (z_scores > threshold).any(axis=1)
        return outlier_mask
        
    def remove_outliers(self, method='zscore', threshold=3):
        if method == 'zscore':
            outlier_mask = self.detect_outliers_zscore(threshold)
            self.df = self.df[~outlier_mask]
        return self
        
    def normalize_data(self, method='minmax'):
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        if method == 'minmax':
            for col in numeric_cols:
                min_val = self.df[col].min()
                max_val = self.df[col].max()
                if max_val > min_val:
                    self.df[col] = (self.df[col] - min_val) / (max_val - min_val)
                    
        elif method == 'standard':
            for col in numeric_cols:
                mean_val = self.df[col].mean()
                std_val = self.df[col].std()
                if std_val > 0:
                    self.df[col] = (self.df[col] - mean_val) / std_val
                    
        return self
        
    def get_cleaned_data(self):
        return self.df
        
    def get_cleaning_report(self):
        rows_removed = self.original_shape[0] - self.df.shape[0]
        cols_removed = self.original_shape[1] - self.df.shape[1]
        
        report = {
            'original_shape': self.original_shape,
            'cleaned_shape': self.df.shape,
            'rows_removed': rows_removed,
            'columns_removed': cols_removed,
            'remaining_rows_percent': (self.df.shape[0] / self.original_shape[0]) * 100
        }
        return report

def clean_dataset(df, remove_duplicates=True, handle_missing=True, 
                  remove_outliers=True, normalize=True):
    cleaner = DataCleaner(df)
    
    if remove_duplicates:
        cleaner.remove_duplicates()
        
    if handle_missing:
        cleaner.handle_missing_values(strategy='mean')
        
    if remove_outliers:
        cleaner.remove_outliers(threshold=3)
        
    if normalize:
        cleaner.normalize_data(method='minmax')
        
    return cleaner.get_cleaned_data(), cleaner.get_cleaning_report()
import json
import re

def clean_data(input_file, output_file, key_to_check='email'):
    """
    Reads a JSON file, filters out entries where the specified key
    does not contain a valid email address, and writes the cleaned data
    to a new JSON file.
    """
    try:
        with open(input_file, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: Input file '{input_file}' is not valid JSON.")
        return

    if not isinstance(data, list):
        print("Error: JSON data is not a list of entries.")
        return

    email_pattern = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
    cleaned_data = []

    for entry in data:
        if not isinstance(entry, dict):
            continue
        value = entry.get(key_to_check)
        if isinstance(value, str) and email_pattern.match(value):
            cleaned_data.append(entry)

    try:
        with open(output_file, 'w') as f:
            json.dump(cleaned_data, f, indent=2)
        print(f"Successfully cleaned data. Valid entries: {len(cleaned_data)}")
        print(f"Output written to: {output_file}")
    except IOError as e:
        print(f"Error writing to output file: {e}")
import numpy as np

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a dataset using the Interquartile Range (IQR) method.
    
    Parameters:
    data (numpy.ndarray): Input data array.
    column (int): Column index to process for outlier removal.
    
    Returns:
    numpy.ndarray: Data with outliers removed.
    """
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a numpy array")
    
    if column >= data.shape[1]:
        raise IndexError("Column index out of bounds")
    
    column_data = data[:, column]
    q1 = np.percentile(column_data, 25)
    q3 = np.percentile(column_data, 75)
    iqr = q3 - q1
    
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    mask = (column_data >= lower_bound) & (column_data <= upper_bound)
    return data[mask]

def calculate_statistics(data):
    """
    Calculate basic statistics for the data.
    
    Parameters:
    data (numpy.ndarray): Input data array.
    
    Returns:
    dict: Dictionary containing mean, median, and standard deviation.
    """
    if data.size == 0:
        return {"mean": np.nan, "median": np.nan, "std": np.nan}
    
    return {
        "mean": np.mean(data),
        "median": np.median(data),
        "std": np.std(data)
    }

def validate_data(data):
    """
    Validate data for cleaning operations.
    
    Parameters:
    data (numpy.ndarray): Input data array.
    
    Returns:
    bool: True if data is valid, False otherwise.
    """
    if data is None:
        return False
    if not isinstance(data, np.ndarray):
        return False
    if data.size == 0:
        return False
    if np.any(np.isnan(data)):
        return False
    return True

def clean_dataset(data, column_index):
    """
    Main function to clean dataset by removing outliers.
    
    Parameters:
    data (numpy.ndarray): Input data array.
    column_index (int): Column index to process.
    
    Returns:
    tuple: (cleaned_data, removed_count, statistics)
    """
    if not validate_data(data):
        raise ValueError("Invalid input data")
    
    original_count = len(data)
    cleaned_data = remove_outliers_iqr(data, column_index)
    removed_count = original_count - len(cleaned_data)
    
    stats = calculate_statistics(cleaned_data[:, column_index])
    
    return cleaned_data, removed_count, stats