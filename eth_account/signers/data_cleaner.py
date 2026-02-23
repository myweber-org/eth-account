
import numpy as np
import pandas as pd
from scipy import stats

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    def remove_outliers_zscore(self, threshold=3):
        df_clean = self.df.copy()
        for col in self.numeric_columns:
            z_scores = np.abs(stats.zscore(df_clean[col].dropna()))
            df_clean = df_clean[(z_scores < threshold) | df_clean[col].isna()]
        return df_clean
    
    def normalize_minmax(self, columns=None):
        if columns is None:
            columns = self.numeric_columns
        
        df_normalized = self.df.copy()
        for col in columns:
            if col in self.numeric_columns:
                col_min = df_normalized[col].min()
                col_max = df_normalized[col].max()
                if col_max > col_min:
                    df_normalized[col] = (df_normalized[col] - col_min) / (col_max - col_min)
        return df_normalized
    
    def fill_missing_median(self, columns=None):
        if columns is None:
            columns = self.numeric_columns
        
        df_filled = self.df.copy()
        for col in columns:
            if col in self.numeric_columns:
                median_val = df_filled[col].median()
                df_filled[col] = df_filled[col].fillna(median_val)
        return df_filled
    
    def get_summary(self):
        summary = {
            'original_shape': self.df.shape,
            'numeric_columns': self.numeric_columns,
            'missing_values': self.df[self.numeric_columns].isnull().sum().to_dict(),
            'data_types': self.df.dtypes.to_dict()
        }
        return summary

def clean_dataset(df, outlier_threshold=3, normalize=True, fill_missing=True):
    cleaner = DataCleaner(df)
    
    if fill_missing:
        df = cleaner.fill_missing_median()
    
    df = cleaner.remove_outliers_zscore(threshold=outlier_threshold)
    
    if normalize:
        df = cleaner.normalize_minmax()
    
    return df