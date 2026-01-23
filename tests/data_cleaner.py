
import pandas as pd
import numpy as np
from scipy import stats

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.numeric_columns = self.df.select_dtypes(include=[np.number]).columns
        self.categorical_columns = self.df.select_dtypes(exclude=[np.number]).columns
    
    def handle_missing_values(self, strategy='mean', fill_value=None):
        if strategy == 'mean':
            self.df[self.numeric_columns] = self.df[self.numeric_columns].fillna(
                self.df[self.numeric_columns].mean()
            )
        elif strategy == 'median':
            self.df[self.numeric_columns] = self.df[self.numeric_columns].fillna(
                self.df[self.numeric_columns].median()
            )
        elif strategy == 'mode':
            for col in self.categorical_columns:
                self.df[col] = self.df[col].fillna(self.df[col].mode()[0] if not self.df[col].mode().empty else 'Unknown')
        elif strategy == 'custom' and fill_value is not None:
            self.df = self.df.fillna(fill_value)
        return self
    
    def remove_outliers_zscore(self, threshold=3):
        z_scores = np.abs(stats.zscore(self.df[self.numeric_columns]))
        outlier_mask = (z_scores < threshold).all(axis=1)
        self.df = self.df[outlier_mask]
        return self
    
    def remove_outliers_iqr(self, multiplier=1.5):
        for col in self.numeric_columns:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - multiplier * IQR
            upper_bound = Q3 + multiplier * IQR
            self.df = self.df[(self.df[col] >= lower_bound) & (self.df[col] <= upper_bound)]
        return self
    
    def standardize_data(self):
        self.df[self.numeric_columns] = (
            self.df[self.numeric_columns] - self.df[self.numeric_columns].mean()
        ) / self.df[self.numeric_columns].std()
        return self
    
    def normalize_data(self):
        self.df[self.numeric_columns] = (
            self.df[self.numeric_columns] - self.df[self.numeric_columns].min()
        ) / (self.df[self.numeric_columns].max() - self.df[self.numeric_columns].min())
        return self
    
    def get_cleaned_data(self):
        return self.df.copy()

def clean_dataset(df, missing_strategy='mean', outlier_method='zscore', standardize=False):
    cleaner = DataCleaner(df)
    cleaner.handle_missing_values(strategy=missing_strategy)
    
    if outlier_method == 'zscore':
        cleaner.remove_outliers_zscore()
    elif outlier_method == 'iqr':
        cleaner.remove_outliers_iqr()
    
    if standardize:
        cleaner.standardize_data()
    
    return cleaner.get_cleaned_data()