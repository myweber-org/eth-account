import numpy as np
import pandas as pd

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.original_shape = df.shape
        
    def remove_outliers_iqr(self, column, multiplier=1.5):
        if column not in self.df.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame")
            
        Q1 = self.df[column].quantile(0.25)
        Q3 = self.df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        
        mask = (self.df[column] >= lower_bound) & (self.df[column] <= upper_bound)
        self.df = self.df[mask]
        removed_count = self.original_shape[0] - self.df.shape[0]
        return removed_count
    
    def normalize_column(self, column, method='minmax'):
        if column not in self.df.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame")
            
        if method == 'minmax':
            min_val = self.df[column].min()
            max_val = self.df[column].max()
            if max_val != min_val:
                self.df[column] = (self.df[column] - min_val) / (max_val - min_val)
            else:
                self.df[column] = 0
                
        elif method == 'zscore':
            mean_val = self.df[column].mean()
            std_val = self.df[column].std()
            if std_val > 0:
                self.df[column] = (self.df[column] - mean_val) / std_val
            else:
                self.df[column] = 0
                
        else:
            raise ValueError("Method must be 'minmax' or 'zscore'")
            
        return self.df[column]
    
    def fill_missing(self, column, strategy='mean'):
        if column not in self.df.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame")
            
        missing_count = self.df[column].isnull().sum()
        
        if strategy == 'mean':
            fill_value = self.df[column].mean()
        elif strategy == 'median':
            fill_value = self.df[column].median()
        elif strategy == 'mode':
            fill_value = self.df[column].mode()[0]
        elif isinstance(strategy, (int, float)):
            fill_value = strategy
        else:
            raise ValueError("Strategy must be 'mean', 'median', 'mode', or a numeric value")
            
        self.df[column] = self.df[column].fillna(fill_value)
        return missing_count
    
    def get_cleaned_data(self):
        return self.df.copy()
    
    def get_summary(self):
        summary = {
            'original_rows': self.original_shape[0],
            'current_rows': self.df.shape[0],
            'columns': list(self.df.columns),
            'missing_values': self.df.isnull().sum().to_dict(),
            'data_types': self.df.dtypes.to_dict()
        }
        return summary

def create_sample_data():
    np.random.seed(42)
    data = {
        'age': np.random.normal(35, 10, 100),
        'income': np.random.exponential(50000, 100),
        'score': np.random.uniform(0, 100, 100)
    }
    
    df = pd.DataFrame(data)
    df.loc[np.random.choice(100, 5), 'age'] = np.nan
    df.loc[np.random.choice(100, 3), 'income'] = np.nan
    
    return df

if __name__ == "__main__":
    sample_df = create_sample_data()
    cleaner = DataCleaner(sample_df)
    
    print("Original data shape:", cleaner.original_shape)
    print("\nMissing values before cleaning:")
    print(sample_df.isnull().sum())
    
    removed = cleaner.remove_outliers_iqr('income')
    print(f"\nRemoved {removed} outliers from 'income'")
    
    cleaner.fill_missing('age', 'mean')
    cleaner.fill_missing('income', 'median')
    
    cleaner.normalize_column('score', 'minmax')
    cleaner.normalize_column('income', 'zscore')
    
    cleaned_df = cleaner.get_cleaned_data()
    print("\nCleaned data shape:", cleaned_df.shape)
    print("\nMissing values after cleaning:")
    print(cleaned_df.isnull().sum())
    
    summary = cleaner.get_summary()
    print(f"\nRows removed: {summary['original_rows'] - summary['current_rows']}")