
import numpy as np
import pandas as pd
from scipy import stats

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.numeric_columns = df.select_dtypes(include=[np.number]).columns
        
    def remove_outliers_iqr(self, column, multiplier=1.5):
        if column not in self.numeric_columns:
            return self.df
        
        Q1 = self.df[column].quantile(0.25)
        Q3 = self.df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        
        return self.df[(self.df[column] >= lower_bound) & (self.df[column] <= upper_bound)]
    
    def remove_outliers_zscore(self, column, threshold=3):
        if column not in self.numeric_columns:
            return self.df
        
        z_scores = np.abs(stats.zscore(self.df[column]))
        return self.df[z_scores < threshold]
    
    def normalize_minmax(self, column):
        if column not in self.numeric_columns:
            return self.df
        
        min_val = self.df[column].min()
        max_val = self.df[column].max()
        self.df[column + '_normalized'] = (self.df[column] - min_val) / (max_val - min_val)
        return self.df
    
    def standardize_zscore(self, column):
        if column not in self.numeric_columns:
            return self.df
        
        mean_val = self.df[column].mean()
        std_val = self.df[column].std()
        self.df[column + '_standardized'] = (self.df[column] - mean_val) / std_val
        return self.df
    
    def handle_missing_mean(self, column):
        if column not in self.numeric_columns:
            return self.df
        
        mean_val = self.df[column].mean()
        self.df[column].fillna(mean_val, inplace=True)
        return self.df
    
    def get_summary(self):
        summary = {
            'total_rows': len(self.df),
            'total_columns': len(self.df.columns),
            'numeric_columns': list(self.numeric_columns),
            'missing_values': self.df.isnull().sum().to_dict(),
            'data_types': self.df.dtypes.to_dict()
        }
        return summary