
import numpy as np
import pandas as pd
from scipy import stats

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.original_shape = df.shape
        
    def remove_outliers_iqr(self, columns=None, threshold=1.5):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
            
        df_clean = self.df.copy()
        for col in columns:
            if col in self.df.columns and self.df[col].dtype in [np.float64, np.int64]:
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
        
        self.df = df_clean
        removed_count = self.original_shape[0] - self.df.shape[0]
        print(f"Removed {removed_count} outliers using IQR method")
        return self
        
    def normalize_minmax(self, columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
            
        for col in columns:
            if col in self.df.columns and self.df[col].dtype in [np.float64, np.int64]:
                min_val = self.df[col].min()
                max_val = self.df[col].max()
                if max_val > min_val:
                    self.df[col] = (self.df[col] - min_val) / (max_val - min_val)
        
        print("Applied Min-Max normalization")
        return self
        
    def standardize_zscore(self, columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
            
        for col in columns:
            if col in self.df.columns and self.df[col].dtype in [np.float64, np.int64]:
                mean_val = self.df[col].mean()
                std_val = self.df[col].std()
                if std_val > 0:
                    self.df[col] = (self.df[col] - mean_val) / std_val
        
        print("Applied Z-score standardization")
        return self
        
    def fill_missing_median(self, columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
            
        for col in columns:
            if col in self.df.columns and self.df[col].dtype in [np.float64, np.int64]:
                if self.df[col].isnull().any():
                    median_val = self.df[col].median()
                    self.df[col].fillna(median_val, inplace=True)
                    print(f"Filled missing values in {col} with median: {median_val}")
        
        return self
        
    def get_cleaned_data(self):
        return self.df
        
    def summary(self):
        print(f"Original shape: {self.original_shape}")
        print(f"Cleaned shape: {self.df.shape}")
        print(f"Removed rows: {self.original_shape[0] - self.df.shape[0]}")
        print(f"Removed columns: {self.original_shape[1] - self.df.shape[1]}")
        print("\nData types:")
        print(self.df.dtypes.value_counts())
        print("\nMissing values:")
        print(self.df.isnull().sum())

def create_sample_data():
    np.random.seed(42)
    data = {
        'feature_a': np.random.normal(100, 15, 1000),
        'feature_b': np.random.exponential(50, 1000),
        'feature_c': np.random.uniform(0, 1, 1000),
        'category': np.random.choice(['A', 'B', 'C'], 1000)
    }
    
    df = pd.DataFrame(data)
    df.loc[np.random.choice(df.index, 50), 'feature_a'] = np.nan
    df.loc[np.random.choice(df.index, 20), 'feature_b'] = 1000
    
    return df

if __name__ == "__main__":
    sample_df = create_sample_data()
    print("Sample data created")
    print(sample_df.head())
    
    cleaner = DataCleaner(sample_df)
    cleaner.summary()
    
    cleaned_df = (cleaner
                 .fill_missing_median()
                 .remove_outliers_iqr(['feature_a', 'feature_b'])
                 .normalize_minmax(['feature_c'])
                 .standardize_zscore(['feature_a', 'feature_b'])
                 .get_cleaned_data())
    
    print("\nCleaning process completed")
    print(cleaned_df.head())