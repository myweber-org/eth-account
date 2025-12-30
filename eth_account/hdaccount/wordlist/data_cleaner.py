
import numpy as np
import pandas as pd
from scipy import stats

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.original_shape = df.shape
        
    def remove_outliers_iqr(self, column, threshold=1.5):
        Q1 = self.df[column].quantile(0.25)
        Q3 = self.df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        mask = (self.df[column] >= lower_bound) & (self.df[column] <= upper_bound)
        self.df = self.df[mask]
        return self
        
    def remove_outliers_zscore(self, column, threshold=3):
        z_scores = np.abs(stats.zscore(self.df[column]))
        mask = z_scores < threshold
        self.df = self.df[mask]
        return self
        
    def normalize_minmax(self, columns):
        for col in columns:
            if col in self.df.columns:
                min_val = self.df[col].min()
                max_val = self.df[col].max()
                if max_val != min_val:
                    self.df[col] = (self.df[col] - min_val) / (max_val - min_val)
        return self
        
    def normalize_zscore(self, columns):
        for col in columns:
            if col in self.df.columns:
                mean_val = self.df[col].mean()
                std_val = self.df[col].std()
                if std_val > 0:
                    self.df[col] = (self.df[col] - mean_val) / std_val
        return self
        
    def fill_missing_mean(self, columns):
        for col in columns:
            if col in self.df.columns:
                self.df[col].fillna(self.df[col].mean(), inplace=True)
        return self
        
    def fill_missing_median(self, columns):
        for col in columns:
            if col in self.df.columns:
                self.df[col].fillna(self.df[col].median(), inplace=True)
        return self
        
    def get_cleaned_data(self):
        return self.df
        
    def get_removed_count(self):
        return self.original_shape[0] - self.df.shape[0]
        
    def get_summary(self):
        summary = {
            'original_rows': self.original_shape[0],
            'cleaned_rows': self.df.shape[0],
            'rows_removed': self.get_removed_count(),
            'removal_percentage': round((self.get_removed_count() / self.original_shape[0]) * 100, 2)
        }
        return summary

def create_sample_data():
    np.random.seed(42)
    data = {
        'feature_a': np.random.normal(100, 15, 1000),
        'feature_b': np.random.exponential(50, 1000),
        'feature_c': np.random.uniform(0, 200, 1000)
    }
    df = pd.DataFrame(data)
    df.iloc[10:20, 0] = np.nan
    df.iloc[50:60, 1] = np.nan
    df.iloc[100:110, 2] = np.nan
    return df

if __name__ == "__main__":
    sample_df = create_sample_data()
    cleaner = DataCleaner(sample_df)
    
    cleaner.fill_missing_mean(['feature_a']) \
           .fill_missing_median(['feature_b', 'feature_c']) \
           .remove_outliers_iqr('feature_a') \
           .remove_outliers_zscore('feature_b') \
           .normalize_minmax(['feature_a', 'feature_c']) \
           .normalize_zscore(['feature_b'])
    
    cleaned_df = cleaner.get_cleaned_data()
    summary = cleaner.get_summary()
    
    print(f"Data cleaning completed:")
    print(f"Original rows: {summary['original_rows']}")
    print(f"Cleaned rows: {summary['cleaned_rows']}")
    print(f"Rows removed: {summary['rows_removed']}")
    print(f"Removal percentage: {summary['removal_percentage']}%")
    print(f"\nCleaned data shape: {cleaned_df.shape}")
    print(f"\nFirst 5 rows of cleaned data:")
    print(cleaned_df.head())
def remove_duplicates_preserve_order(sequence):
    seen = set()
    result = []
    for item in sequence:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result