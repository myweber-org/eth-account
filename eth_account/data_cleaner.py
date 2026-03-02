
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
            if col in self.df.columns and pd.api.types.is_numeric_dtype(self.df[col]):
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
        
        self.df = df_clean
        removed_count = self.original_shape[0] - self.df.shape[0]
        return removed_count
    
    def normalize_column(self, column, method='minmax'):
        if column not in self.df.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame")
            
        if method == 'minmax':
            min_val = self.df[column].min()
            max_val = self.df[column].max()
            if max_val - min_val > 0:
                self.df[f'{column}_normalized'] = (self.df[column] - min_val) / (max_val - min_val)
            else:
                self.df[f'{column}_normalized'] = 0
                
        elif method == 'zscore':
            mean_val = self.df[column].mean()
            std_val = self.df[column].std()
            if std_val > 0:
                self.df[f'{column}_normalized'] = (self.df[column] - mean_val) / std_val
            else:
                self.df[f'{column}_normalized'] = 0
                
        return self.df[f'{column}_normalized']
    
    def fill_missing_values(self, strategy='mean', columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
            
        for col in columns:
            if col in self.df.columns and self.df[col].isnull().any():
                if strategy == 'mean':
                    fill_value = self.df[col].mean()
                elif strategy == 'median':
                    fill_value = self.df[col].median()
                elif strategy == 'mode':
                    fill_value = self.df[col].mode()[0] if not self.df[col].mode().empty else 0
                else:
                    fill_value = 0
                    
                self.df[col].fillna(fill_value, inplace=True)
                
        return self.df.isnull().sum().sum()
    
    def get_clean_data(self):
        return self.df.copy()
    
    def get_summary(self):
        summary = {
            'original_rows': self.original_shape[0],
            'current_rows': self.df.shape[0],
            'columns': self.df.shape[1],
            'missing_values': self.df.isnull().sum().sum(),
            'numeric_columns': list(self.df.select_dtypes(include=[np.number]).columns)
        }
        return summary

def create_sample_data():
    np.random.seed(42)
    data = {
        'feature_a': np.random.normal(100, 15, 1000),
        'feature_b': np.random.exponential(50, 1000),
        'feature_c': np.random.randint(1, 100, 1000),
        'category': np.random.choice(['A', 'B', 'C'], 1000)
    }
    
    df = pd.DataFrame(data)
    df.loc[np.random.choice(df.index, 50), 'feature_a'] = np.nan
    df.loc[np.random.choice(df.index, 30), 'feature_b'] = np.nan
    
    outliers = np.random.randint(0, 1000, 20)
    df.loc[outliers, 'feature_c'] = np.random.randint(200, 500, 20)
    
    return df

if __name__ == "__main__":
    sample_df = create_sample_data()
    cleaner = DataCleaner(sample_df)
    
    print("Initial summary:")
    print(cleaner.get_summary())
    
    removed = cleaner.remove_outliers_iqr(['feature_c'])
    print(f"\nRemoved {removed} outliers from feature_c")
    
    missing_filled = cleaner.fill_missing_values(strategy='mean')
    print(f"Filled {missing_filled} missing values")
    
    cleaner.normalize_column('feature_a', method='zscore')
    cleaner.normalize_column('feature_b', method='minmax')
    
    print("\nFinal summary:")
    print(cleaner.get_summary())
    
    clean_data = cleaner.get_clean_data()
    print(f"\nClean data shape: {clean_data.shape}")
    print("First 5 rows of normalized columns:")
    print(clean_data[['feature_a_normalized', 'feature_b_normalized']].head())