
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
                mask = (self.df[col] >= lower_bound) & (self.df[col] <= upper_bound)
                df_clean = df_clean[mask]
        
        removed_count = self.original_shape[0] - df_clean.shape[0]
        self.df = df_clean.reset_index(drop=True)
        return removed_count
    
    def normalize_column(self, column, method='minmax'):
        if column not in self.df.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame")
            
        if not pd.api.types.is_numeric_dtype(self.df[column]):
            raise ValueError(f"Column '{column}' is not numeric")
            
        if method == 'minmax':
            min_val = self.df[column].min()
            max_val = self.df[column].max()
            if max_val == min_val:
                self.df[f'{column}_normalized'] = 0.5
            else:
                self.df[f'{column}_normalized'] = (self.df[column] - min_val) / (max_val - min_val)
                
        elif method == 'zscore':
            mean_val = self.df[column].mean()
            std_val = self.df[column].std()
            if std_val == 0:
                self.df[f'{column}_normalized'] = 0
            else:
                self.df[f'{column}_normalized'] = (self.df[column] - mean_val) / std_val
                
        else:
            raise ValueError(f"Unknown normalization method: {method}")
            
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
                    fill_value = self.df[col].mode()[0]
                else:
                    raise ValueError(f"Unknown strategy: {strategy}")
                    
                self.df[col] = self.df[col].fillna(fill_value)
                
        return self.df.isnull().sum().sum()
    
    def get_clean_data(self):
        return self.df
    
    def get_summary(self):
        summary = {
            'original_rows': self.original_shape[0],
            'current_rows': self.df.shape[0],
            'columns': self.df.shape[1],
            'missing_values': self.df.isnull().sum().sum(),
            'numeric_columns': list(self.df.select_dtypes(include=[np.number]).columns),
            'categorical_columns': list(self.df.select_dtypes(include=['object']).columns)
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
    
    df.loc[10, 'income'] = 1000000
    df.loc[20, 'age'] = 150
    
    return df

if __name__ == "__main__":
    sample_df = create_sample_data()
    print("Sample data created with shape:", sample_df.shape)
    
    cleaner = DataCleaner(sample_df)
    print("\nInitial summary:")
    print(cleaner.get_summary())
    
    removed = cleaner.remove_outliers_iqr(['age', 'income'])
    print(f"\nRemoved {removed} outliers using IQR method")
    
    missing_filled = cleaner.fill_missing_values(strategy='median')
    print(f"Filled {missing_filled} missing values")
    
    cleaner.normalize_column('score', method='minmax')
    cleaner.normalize_column('income', method='zscore')
    
    print("\nFinal summary:")
    print(cleaner.get_summary())
    
    clean_data = cleaner.get_clean_data()
    print(f"\nCleaned data shape: {clean_data.shape}")
    print("\nFirst 5 rows of cleaned data:")
    print(clean_data.head())import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def normalize_minmax(df, column):
    min_val = df[column].min()
    max_val = df[column].max()
    df[column + '_normalized'] = (df[column] - min_val) / (max_val - min_val)
    return df

def standardize_zscore(df, column):
    mean_val = df[column].mean()
    std_val = df[column].std()
    df[column + '_standardized'] = (df[column] - mean_val) / std_val
    return df

def handle_missing_mean(df, column):
    mean_val = df[column].mean()
    df[column].fillna(mean_val, inplace=True)
    return df

def clean_dataset(df, numeric_columns):
    cleaned_df = df.copy()
    for col in numeric_columns:
        if cleaned_df[col].isnull().sum() > 0:
            cleaned_df = handle_missing_mean(cleaned_df, col)
        cleaned_df = remove_outliers_iqr(cleaned_df, col)
        cleaned_df = normalize_minmax(cleaned_df, col)
        cleaned_df = standardize_zscore(cleaned_df, col)
    return cleaned_df

if __name__ == "__main__":
    sample_data = {'A': [1, 2, 3, 4, 5, 100],
                   'B': [10, 20, 30, 40, 50, 200],
                   'C': [0.1, 0.2, 0.3, 0.4, 0.5, 5.0]}
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    
    cleaned = clean_dataset(df, ['A', 'B', 'C'])
    print("\nCleaned DataFrame:")
    print(cleaned)