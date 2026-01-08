import numpy as np
import pandas as pd
from scipy import stats

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.original_shape = df.shape
        
    def remove_outliers_iqr(self, columns=None, factor=1.5):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
            
        df_clean = self.df.copy()
        for col in columns:
            if col in self.df.columns:
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - factor * IQR
                upper_bound = Q3 + factor * IQR
                mask = (self.df[col] >= lower_bound) & (self.df[col] <= upper_bound)
                df_clean = df_clean[mask]
        return df_clean
    
    def normalize_minmax(self, columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
            
        df_normalized = self.df.copy()
        for col in columns:
            if col in self.df.columns:
                min_val = self.df[col].min()
                max_val = self.df[col].max()
                if max_val > min_val:
                    df_normalized[col] = (self.df[col] - min_val) / (max_val - min_val)
        return df_normalized
    
    def standardize_zscore(self, columns=None, threshold=3):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
            
        df_standardized = self.df.copy()
        for col in columns:
            if col in self.df.columns:
                mean_val = self.df[col].mean()
                std_val = self.df[col].std()
                if std_val > 0:
                    z_scores = (self.df[col] - mean_val) / std_val
                    df_standardized[col] = z_scores
                    if threshold:
                        df_standardized = df_standardized[abs(z_scores) <= threshold]
        return df_standardized
    
    def handle_missing(self, strategy='mean', fill_value=None):
        df_filled = self.df.copy()
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if self.df[col].isnull().any():
                if strategy == 'mean':
                    fill_val = self.df[col].mean()
                elif strategy == 'median':
                    fill_val = self.df[col].median()
                elif strategy == 'mode':
                    fill_val = self.df[col].mode()[0]
                elif strategy == 'constant' and fill_value is not None:
                    fill_val = fill_value
                else:
                    fill_val = 0
                df_filled[col] = self.df[col].fillna(fill_val)
        return df_filled
    
    def get_summary(self):
        summary = {
            'original_rows': self.original_shape[0],
            'original_columns': self.original_shape[1],
            'numeric_columns': list(self.df.select_dtypes(include=[np.number]).columns),
            'missing_values': self.df.isnull().sum().sum(),
            'missing_percentage': (self.df.isnull().sum().sum() / (self.original_shape[0] * self.original_shape[1])) * 100
        }
        return summary

def create_sample_data():
    np.random.seed(42)
    data = {
        'feature_a': np.random.normal(100, 15, 100),
        'feature_b': np.random.exponential(50, 100),
        'feature_c': np.random.uniform(0, 1, 100),
        'category': np.random.choice(['A', 'B', 'C'], 100)
    }
    df = pd.DataFrame(data)
    df.iloc[5:10, 0] = np.nan
    df.iloc[15:20, 1] = np.nan
    df.loc[95, 'feature_a'] = 500
    df.loc[96, 'feature_b'] = 300
    return df

if __name__ == "__main__":
    sample_df = create_sample_data()
    cleaner = DataCleaner(sample_df)
    
    print("Data Summary:")
    summary = cleaner.get_summary()
    for key, value in summary.items():
        print(f"{key}: {value}")
    
    cleaned_df = cleaner.remove_outliers_iqr()
    normalized_df = cleaner.normalize_minmax()
    standardized_df = cleaner.standardize_zscore()
    filled_df = cleaner.handle_missing(strategy='mean')
    
    print(f"\nOriginal shape: {sample_df.shape}")
    print(f"Cleaned shape: {cleaned_df.shape}")
    print(f"Normalized shape: {normalized_df.shape}")
    print(f"Standardized shape: {standardized_df.shape}")
    print(f"Filled shape: {filled_df.shape}")