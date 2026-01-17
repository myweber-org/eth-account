
import numpy as np
import pandas as pd
from scipy import stats

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.original_df = df.copy()
        
    def remove_outliers_iqr(self, columns=None, factor=1.5):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        cleaned_df = self.df.copy()
        
        for col in columns:
            if col in cleaned_df.columns:
                Q1 = cleaned_df[col].quantile(0.25)
                Q3 = cleaned_df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - factor * IQR
                upper_bound = Q3 + factor * IQR
                
                mask = (cleaned_df[col] >= lower_bound) & (cleaned_df[col] <= upper_bound)
                cleaned_df = cleaned_df[mask]
        
        self.df = cleaned_df
        return self
    
    def remove_outliers_zscore(self, columns=None, threshold=3):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        cleaned_df = self.df.copy()
        
        for col in columns:
            if col in cleaned_df.columns:
                z_scores = np.abs(stats.zscore(cleaned_df[col].dropna()))
                mask = z_scores < threshold
                cleaned_df = cleaned_df[mask]
        
        self.df = cleaned_df
        return self
    
    def normalize_minmax(self, columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        normalized_df = self.df.copy()
        
        for col in columns:
            if col in normalized_df.columns:
                min_val = normalized_df[col].min()
                max_val = normalized_df[col].max()
                
                if max_val != min_val:
                    normalized_df[col] = (normalized_df[col] - min_val) / (max_val - min_val)
        
        self.df = normalized_df
        return self
    
    def normalize_zscore(self, columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        normalized_df = self.df.copy()
        
        for col in columns:
            if col in normalized_df.columns:
                mean_val = normalized_df[col].mean()
                std_val = normalized_df[col].std()
                
                if std_val != 0:
                    normalized_df[col] = (normalized_df[col] - mean_val) / std_val
        
        self.df = normalized_df
        return self
    
    def fill_missing_mean(self, columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        filled_df = self.df.copy()
        
        for col in columns:
            if col in filled_df.columns:
                filled_df[col] = filled_df[col].fillna(filled_df[col].mean())
        
        self.df = filled_df
        return self
    
    def fill_missing_median(self, columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        filled_df = self.df.copy()
        
        for col in columns:
            if col in filled_df.columns:
                filled_df[col] = filled_df[col].fillna(filled_df[col].median())
        
        self.df = filled_df
        return self
    
    def get_cleaned_data(self):
        return self.df
    
    def get_removed_count(self):
        return len(self.original_df) - len(self.df)
    
    def reset(self):
        self.df = self.original_df.copy()
        return self

def load_sample_data():
    data = {
        'feature_a': np.random.normal(100, 15, 1000),
        'feature_b': np.random.exponential(50, 1000),
        'feature_c': np.random.uniform(0, 200, 1000),
        'category': np.random.choice(['A', 'B', 'C'], 1000)
    }
    
    data['feature_a'][np.random.choice(1000, 50)] = np.nan
    data['feature_b'][np.random.choice(1000, 30)] = np.nan
    
    outliers = np.random.choice(1000, 20)
    data['feature_a'][outliers] = np.random.uniform(300, 500, 20)
    data['feature_b'][outliers] = np.random.uniform(300, 500, 20)
    
    return pd.DataFrame(data)

if __name__ == "__main__":
    df = load_sample_data()
    print(f"Original data shape: {df.shape}")
    
    cleaner = DataCleaner(df)
    
    cleaner.remove_outliers_iqr(['feature_a', 'feature_b', 'feature_c'])
    cleaner.fill_missing_mean(['feature_a', 'feature_b'])
    cleaner.normalize_minmax(['feature_a', 'feature_b', 'feature_c'])
    
    cleaned_df = cleaner.get_cleaned_data()
    print(f"Cleaned data shape: {cleaned_df.shape}")
    print(f"Removed {cleaner.get_removed_count()} rows")
    
    print("\nSummary statistics:")
    print(cleaned_df.describe())