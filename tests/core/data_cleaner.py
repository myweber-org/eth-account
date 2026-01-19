
import numpy as np
import pandas as pd
from scipy import stats

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.original_df = df.copy()
        
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
    
    def normalize_minmax(self, column):
        min_val = self.df[column].min()
        max_val = self.df[column].max()
        self.df[column] = (self.df[column] - min_val) / (max_val - min_val)
        return self
    
    def normalize_zscore(self, column):
        mean_val = self.df[column].mean()
        std_val = self.df[column].std()
        self.df[column] = (self.df[column] - mean_val) / std_val
        return self
    
    def fill_missing_mean(self, column):
        self.df[column].fillna(self.df[column].mean(), inplace=True)
        return self
    
    def fill_missing_median(self, column):
        self.df[column].fillna(self.df[column].median(), inplace=True)
        return self
    
    def drop_duplicates(self):
        self.df.drop_duplicates(inplace=True)
        return self
    
    def get_cleaned_data(self):
        return self.df
    
    def reset_to_original(self):
        self.df = self.original_df.copy()
        return self
    
    def summary(self):
        print(f"Original shape: {self.original_df.shape}")
        print(f"Cleaned shape: {self.df.shape}")
        print(f"Rows removed: {self.original_df.shape[0] - self.df.shape[0]}")
        print(f"Columns: {list(self.df.columns)}")
        return self

def example_usage():
    data = {
        'feature1': np.random.normal(100, 15, 1000),
        'feature2': np.random.exponential(50, 1000),
        'feature3': np.random.uniform(0, 1, 1000)
    }
    df = pd.DataFrame(data)
    df.iloc[10, 0] = 500
    df.iloc[20, 1] = 1000
    
    cleaner = DataCleaner(df)
    cleaner.remove_outliers_iqr('feature1') \
           .remove_outliers_zscore('feature2') \
           .normalize_minmax('feature3') \
           .fill_missing_mean('feature1') \
           .drop_duplicates()
    
    cleaned_df = cleaner.get_cleaned_data()
    cleaner.summary()
    
    return cleaned_df

if __name__ == "__main__":
    result = example_usage()
    print(result.head())