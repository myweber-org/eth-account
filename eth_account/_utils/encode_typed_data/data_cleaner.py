import numpy as np
import pandas as pd
from scipy import stats

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.original_shape = df.shape
        
    def detect_outliers_iqr(self, column, threshold=1.5):
        Q1 = self.df[column].quantile(0.25)
        Q3 = self.df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        outliers = self.df[(self.df[column] < lower_bound) | (self.df[column] > upper_bound)]
        return outliers.index.tolist()
    
    def remove_outliers(self, columns, method='iqr', threshold=1.5):
        outlier_indices = []
        for col in columns:
            if method == 'iqr':
                indices = self.detect_outliers_iqr(col, threshold)
                outlier_indices.extend(indices)
        
        unique_outliers = list(set(outlier_indices))
        self.df = self.df.drop(index=unique_outliers)
        return len(unique_outliers)
    
    def normalize_column(self, column, method='zscore'):
        if method == 'zscore':
            self.df[column] = stats.zscore(self.df[column])
        elif method == 'minmax':
            min_val = self.df[column].min()
            max_val = self.df[column].max()
            self.df[column] = (self.df[column] - min_val) / (max_val - min_val)
        return self.df[column]
    
    def fill_missing(self, column, method='mean'):
        if method == 'mean':
            fill_value = self.df[column].mean()
        elif method == 'median':
            fill_value = self.df[column].median()
        elif method == 'mode':
            fill_value = self.df[column].mode()[0]
        else:
            fill_value = method
        
        self.df[column] = self.df[column].fillna(fill_value)
        return self.df[column]
    
    def get_cleaned_data(self):
        return self.df
    
    def get_cleaning_report(self):
        removed_rows = self.original_shape[0] - self.df.shape[0]
        return {
            'original_rows': self.original_shape[0],
            'cleaned_rows': self.df.shape[0],
            'removed_rows': removed_rows,
            'removed_percentage': (removed_rows / self.original_shape[0]) * 100
        }

def example_usage():
    np.random.seed(42)
    data = {
        'feature1': np.random.normal(100, 15, 1000),
        'feature2': np.random.exponential(50, 1000),
        'category': np.random.choice(['A', 'B', 'C'], 1000)
    }
    
    df = pd.DataFrame(data)
    cleaner = DataCleaner(df)
    
    outliers_removed = cleaner.remove_outliers(['feature1', 'feature2'])
    cleaner.normalize_column('feature1', method='zscore')
    cleaner.normalize_column('feature2', method='minmax')
    
    report = cleaner.get_cleaning_report()
    cleaned_df = cleaner.get_cleaned_data()
    
    return cleaned_df, report