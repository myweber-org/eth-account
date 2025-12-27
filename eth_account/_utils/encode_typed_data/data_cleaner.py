
import numpy as np
import pandas as pd
from scipy import stats

class DataCleaner:
    def __init__(self, data):
        self.data = data
        self.original_shape = data.shape
        
    def remove_outliers_iqr(self, columns=None, factor=1.5):
        if columns is None:
            columns = self.data.columns
            
        clean_data = self.data.copy()
        for col in columns:
            if pd.api.types.is_numeric_dtype(clean_data[col]):
                Q1 = clean_data[col].quantile(0.25)
                Q3 = clean_data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - factor * IQR
                upper_bound = Q3 + factor * IQR
                clean_data = clean_data[(clean_data[col] >= lower_bound) & (clean_data[col] <= upper_bound)]
        return clean_data
    
    def remove_outliers_zscore(self, columns=None, threshold=3):
        if columns is None:
            columns = self.data.columns
            
        clean_data = self.data.copy()
        for col in columns:
            if pd.api.types.is_numeric_dtype(clean_data[col]):
                z_scores = np.abs(stats.zscore(clean_data[col]))
                clean_data = clean_data[z_scores < threshold]
        return clean_data
    
    def normalize_minmax(self, columns=None):
        if columns is None:
            columns = self.data.select_dtypes(include=[np.number]).columns
            
        normalized_data = self.data.copy()
        for col in columns:
            if pd.api.types.is_numeric_dtype(normalized_data[col]):
                min_val = normalized_data[col].min()
                max_val = normalized_data[col].max()
                if max_val != min_val:
                    normalized_data[col] = (normalized_data[col] - min_val) / (max_val - min_val)
        return normalized_data
    
    def normalize_zscore(self, columns=None):
        if columns is None:
            columns = self.data.select_dtypes(include=[np.number]).columns
            
        normalized_data = self.data.copy()
        for col in columns:
            if pd.api.types.is_numeric_dtype(normalized_data[col]):
                mean_val = normalized_data[col].mean()
                std_val = normalized_data[col].std()
                if std_val > 0:
                    normalized_data[col] = (normalized_data[col] - mean_val) / std_val
        return normalized_data
    
    def fill_missing_mean(self, columns=None):
        if columns is None:
            columns = self.data.columns
            
        filled_data = self.data.copy()
        for col in columns:
            if pd.api.types.is_numeric_dtype(filled_data[col]):
                filled_data[col] = filled_data[col].fillna(filled_data[col].mean())
        return filled_data
    
    def get_cleaning_report(self):
        report = {
            'original_rows': self.original_shape[0],
            'original_columns': self.original_shape[1],
            'current_rows': self.data.shape[0],
            'current_columns': self.data.shape[1],
            'missing_values': self.data.isnull().sum().sum(),
            'numeric_columns': list(self.data.select_dtypes(include=[np.number]).columns),
            'categorical_columns': list(self.data.select_dtypes(exclude=[np.number]).columns)
        }
        return report

def load_and_clean_data(filepath, outlier_method='iqr', normalization_method='minmax'):
    try:
        data = pd.read_csv(filepath)
        cleaner = DataCleaner(data)
        
        if outlier_method == 'iqr':
            data = cleaner.remove_outliers_iqr()
        elif outlier_method == 'zscore':
            data = cleaner.remove_outliers_zscore()
        
        if normalization_method == 'minmax':
            data = cleaner.normalize_minmax()
        elif normalization_method == 'zscore':
            data = cleaner.normalize_zscore()
        
        data = cleaner.fill_missing_mean()
        report = cleaner.get_cleaning_report()
        
        return data, report
    except Exception as e:
        print(f"Error during data cleaning: {e}")
        return None, None