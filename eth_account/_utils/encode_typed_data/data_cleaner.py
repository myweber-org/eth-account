
import pandas as pd
import numpy as np
from scipy import stats

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.original_shape = df.shape
        
    def remove_outliers_zscore(self, columns=None, threshold=3):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
            
        df_clean = self.df.copy()
        for col in columns:
            if col in self.df.columns:
                z_scores = np.abs(stats.zscore(self.df[col].dropna()))
                mask = z_scores < threshold
                df_clean = df_clean[mask.reindex(df_clean.index, fill_value=True)]
        
        removed_count = len(self.df) - len(df_clean)
        self.df = df_clean
        return removed_count
    
    def normalize_minmax(self, columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
            
        df_normalized = self.df.copy()
        for col in columns:
            if col in self.df.columns:
                col_min = self.df[col].min()
                col_max = self.df[col].max()
                if col_max > col_min:
                    df_normalized[col] = (self.df[col] - col_min) / (col_max - col_min)
        
        self.df = df_normalized
        return self
    
    def fill_missing_median(self, columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
            
        for col in columns:
            if col in self.df.columns and self.df[col].isnull().any():
                median_val = self.df[col].median()
                self.df[col].fillna(median_val, inplace=True)
        
        return self
    
    def get_cleaned_data(self):
        return self.df
    
    def get_summary(self):
        summary = {
            'original_rows': self.original_shape[0],
            'cleaned_rows': len(self.df),
            'removed_rows': self.original_shape[0] - len(self.df),
            'original_columns': self.original_shape[1],
            'cleaned_columns': self.df.shape[1]
        }
        return summary

def load_and_clean_data(filepath, outlier_threshold=3):
    try:
        df = pd.read_csv(filepath)
        cleaner = DataCleaner(df)
        
        print(f"Loaded data: {df.shape[0]} rows, {df.shape[1]} columns")
        
        removed = cleaner.remove_outliers_zscore(threshold=outlier_threshold)
        print(f"Removed {removed} outliers using Z-score method")
        
        cleaner.fill_missing_median()
        print("Filled missing values with median")
        
        cleaner.normalize_minmax()
        print("Applied min-max normalization")
        
        summary = cleaner.get_summary()
        print(f"Cleaning complete. Final shape: {summary['cleaned_rows']}x{summary['cleaned_columns']}")
        
        return cleaner.get_cleaned_data()
        
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None
    except Exception as e:
        print(f"Error during data cleaning: {str(e)}")
        return None

if __name__ == "__main__":
    cleaned_df = load_and_clean_data('sample_data.csv')
    if cleaned_df is not None:
        print(cleaned_df.head())
        print(cleaned_df.describe())