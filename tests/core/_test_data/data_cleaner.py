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
    
    def impute_missing_mean(self, column):
        if column not in self.numeric_columns:
            return self.df
            
        mean_value = self.df[column].mean()
        self.df[column].fillna(mean_value, inplace=True)
        return self.df
    
    def standardize_column(self, column):
        if column not in self.numeric_columns:
            return self.df
            
        mean = self.df[column].mean()
        std = self.df[column].std()
        self.df[column] = (self.df[column] - mean) / std
        return self.df
    
    def detect_skewed_columns(self, threshold=0.5):
        skewed_cols = []
        for col in self.numeric_columns:
            skewness = stats.skew(self.df[col].dropna())
            if abs(skewness) > threshold:
                skewed_cols.append((col, skewness))
        return skewed_cols
    
    def get_summary(self):
        summary = {
            'total_rows': len(self.df),
            'total_columns': len(self.df.columns),
            'missing_values': self.df.isnull().sum().sum(),
            'numeric_columns': list(self.numeric_columns),
            'memory_usage': self.df.memory_usage(deep=True).sum() / 1024**2
        }
        return summary

def process_dataset(file_path):
    df = pd.read_csv(file_path)
    cleaner = DataCleaner(df)
    
    print("Initial summary:")
    print(cleaner.get_summary())
    
    for col in cleaner.numeric_columns:
        cleaner.impute_missing_mean(col)
        cleaner.remove_outliers_iqr(col)
    
    skewed = cleaner.detect_skewed_columns()
    if skewed:
        print(f"Found {len(skewed)} skewed columns")
    
    print("Final summary:")
    print(cleaner.get_summary())
    
    return cleaner.df