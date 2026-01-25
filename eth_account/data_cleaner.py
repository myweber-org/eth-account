
import numpy as np
import pandas as pd
from scipy import stats

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.original_columns = df.columns.tolist()
    
    def remove_outliers_iqr(self, column, multiplier=1.5):
        Q1 = self.df[column].quantile(0.25)
        Q3 = self.df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        self.df = self.df[(self.df[column] >= lower_bound) & (self.df[column] <= upper_bound)]
        return self
    
    def remove_outliers_zscore(self, column, threshold=3):
        z_scores = np.abs(stats.zscore(self.df[column]))
        self.df = self.df[z_scores < threshold]
        return self
    
    def normalize_column(self, column, method='minmax'):
        if method == 'minmax':
            min_val = self.df[column].min()
            max_val = self.df[column].max()
            if max_val > min_val:
                self.df[column] = (self.df[column] - min_val) / (max_val - min_val)
        elif method == 'zscore':
            mean_val = self.df[column].mean()
            std_val = self.df[column].std()
            if std_val > 0:
                self.df[column] = (self.df[column] - mean_val) / std_val
        return self
    
    def fill_missing(self, column, strategy='mean'):
        if strategy == 'mean':
            fill_value = self.df[column].mean()
        elif strategy == 'median':
            fill_value = self.df[column].median()
        elif strategy == 'mode':
            fill_value = self.df[column].mode()[0]
        else:
            fill_value = 0
        
        self.df[column] = self.df[column].fillna(fill_value)
        return self
    
    def get_cleaned_data(self):
        return self.df
    
    def get_summary(self):
        summary = {
            'original_rows': len(self.df),
            'cleaned_rows': len(self.df),
            'columns': self.original_columns,
            'remaining_columns': self.df.columns.tolist()
        }
        return summary

def clean_dataset(df, config):
    cleaner = DataCleaner(df)
    
    for column in config.get('outlier_columns', []):
        method = config.get('outlier_method', 'iqr')
        if method == 'iqr':
            cleaner.remove_outliers_iqr(column)
        elif method == 'zscore':
            cleaner.remove_outliers_zscore(column)
    
    for column in config.get('normalize_columns', []):
        method = config.get('normalize_method', 'minmax')
        cleaner.normalize_column(column, method)
    
    for column in config.get('fill_missing_columns', []):
        strategy = config.get('fill_strategy', 'mean')
        cleaner.fill_missing(column, strategy)
    
    return cleaner.get_cleaned_data()