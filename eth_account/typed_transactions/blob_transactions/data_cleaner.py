
import numpy as np
import pandas as pd
from scipy import stats

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.original_columns = df.columns.tolist()
        
    def remove_outliers_zscore(self, columns=None, threshold=3):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
            
        df_clean = self.df.copy()
        for col in columns:
            if col in self.df.columns and self.df[col].dtype in [np.float64, np.int64]:
                z_scores = np.abs(stats.zscore(self.df[col].dropna()))
                mask = z_scores < threshold
                df_clean = df_clean[mask.reindex(df_clean.index, fill_value=True)]
        
        self.df = df_clean
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
        
        return self
        
    def fill_missing_median(self, columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
            
        for col in columns:
            if col in self.df.columns and self.df[col].dtype in [np.float64, np.int64]:
                median_val = self.df[col].median()
                self.df[col] = self.df[col].fillna(median_val)
        
        return self
        
    def get_cleaned_data(self):
        return self.df
        
    def get_summary(self):
        summary = {
            'original_shape': (len(self.df), len(self.original_columns)),
            'cleaned_shape': self.df.shape,
            'missing_values': self.df.isnull().sum().sum(),
            'numeric_columns': self.df.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical_columns': self.df.select_dtypes(include=['object']).columns.tolist()
        }
        return summary

def create_sample_data():
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    data = {
        'date': dates,
        'temperature': np.random.normal(25, 5, 100),
        'humidity': np.random.normal(60, 15, 100),
        'pressure': np.random.normal(1013, 10, 100),
        'category': np.random.choice(['A', 'B', 'C'], 100)
    }
    
    df = pd.DataFrame(data)
    df.loc[np.random.choice(df.index, 5), 'temperature'] = np.nan
    df.loc[np.random.choice(df.index, 3), 'humidity'] = np.nan
    df.loc[10:15, 'pressure'] = df.loc[10:15, 'pressure'] * 10
    
    return df

if __name__ == "__main__":
    sample_df = create_sample_data()
    print("Original data shape:", sample_df.shape)
    print("\nMissing values before cleaning:")
    print(sample_df.isnull().sum())
    
    cleaner = DataCleaner(sample_df)
    cleaned_df = (cleaner
                 .fill_missing_median()
                 .remove_outliers_zscore(threshold=3)
                 .normalize_minmax()
                 .get_cleaned_data())
    
    print("\nCleaned data shape:", cleaned_df.shape)
    print("\nMissing values after cleaning:")
    print(cleaned_df.isnull().sum())
    
    summary = cleaner.get_summary()
    print("\nCleaning summary:")
    for key, value in summary.items():
        print(f"{key}: {value}")import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a specified column in a DataFrame using the IQR method.
    
    Args:
        df (pd.DataFrame): The input DataFrame.
        column (str): The column name to process.
    
    Returns:
        pd.DataFrame: DataFrame with outliers removed from the specified column.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
    return filtered_df

def calculate_summary_statistics(df):
    """
    Calculate summary statistics for numeric columns in a DataFrame.
    
    Args:
        df (pd.DataFrame): The input DataFrame.
    
    Returns:
        pd.DataFrame: Summary statistics for numeric columns.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    summary = df[numeric_cols].describe()
    
    return summary

def clean_dataset(df, numeric_columns=None):
    """
    Clean a dataset by removing outliers from specified numeric columns.
    
    Args:
        df (pd.DataFrame): The input DataFrame.
        numeric_columns (list, optional): List of numeric column names to clean.
            If None, all numeric columns will be processed.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame with outliers removed.
    """
    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_df = df.copy()
    
    for column in numeric_columns:
        if column in df.columns and pd.api.types.is_numeric_dtype(df[column]):
            cleaned_df = remove_outliers_iqr(cleaned_df, column)
    
    return cleaned_df

if __name__ == "__main__":
    sample_data = {
        'id': range(1, 101),
        'value': np.random.randn(100) * 10 + 50,
        'category': np.random.choice(['A', 'B', 'C'], 100)
    }
    
    df = pd.DataFrame(sample_data)
    df.loc[10, 'value'] = 200
    df.loc[20, 'value'] = -100
    
    print("Original dataset shape:", df.shape)
    print("Original summary statistics:")
    print(calculate_summary_statistics(df))
    
    cleaned_df = clean_dataset(df, ['value'])
    
    print("\nCleaned dataset shape:", cleaned_df.shape)
    print("Cleaned summary statistics:")
    print(calculate_summary_statistics(cleaned_df))
import pandas as pd
import numpy as np

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

def clean_dataset(file_path):
    df = pd.read_csv(file_path)
    
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_columns:
        df = remove_outliers_iqr(df, col)
        df = normalize_minmax(df, col)
    
    return df

if __name__ == "__main__":
    cleaned_data = clean_dataset('raw_data.csv')
    cleaned_data.to_csv('cleaned_data.csv', index=False)
    print(f"Data cleaning complete. Processed {len(cleaned_data)} records.")
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
                
        self.df = df_clean.reset_index(drop=True)
        return self
        
    def normalize_minmax(self, columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
            
        df_normalized = self.df.copy()
        for col in columns:
            if col in self.df.columns and pd.api.types.is_numeric_dtype(self.df[col]):
                min_val = self.df[col].min()
                max_val = self.df[col].max()
                if max_val != min_val:
                    df_normalized[col] = (self.df[col] - min_val) / (max_val - min_val)
                    
        self.df = df_normalized
        return self
        
    def standardize_zscore(self, columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
            
        df_standardized = self.df.copy()
        for col in columns:
            if col in self.df.columns and pd.api.types.is_numeric_dtype(self.df[col]):
                mean_val = self.df[col].mean()
                std_val = self.df[col].std()
                if std_val > 0:
                    df_standardized[col] = (self.df[col] - mean_val) / std_val
                    
        self.df = df_standardized
        return self
        
    def handle_missing_values(self, strategy='mean', columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
            
        df_filled = self.df.copy()
        for col in columns:
            if col in self.df.columns and pd.api.types.is_numeric_dtype(self.df[col]):
                if strategy == 'mean':
                    fill_value = self.df[col].mean()
                elif strategy == 'median':
                    fill_value = self.df[col].median()
                elif strategy == 'mode':
                    fill_value = self.df[col].mode()[0] if not self.df[col].mode().empty else 0
                else:
                    fill_value = 0
                    
                df_filled[col] = self.df[col].fillna(fill_value)
                
        self.df = df_filled
        return self
        
    def get_cleaned_data(self):
        return self.df
        
    def get_removed_count(self):
        return self.original_shape[0] - self.df.shape[0]
        
    def summary(self):
        print(f"Original data shape: {self.original_shape}")
        print(f"Cleaned data shape: {self.df.shape}")
        print(f"Rows removed: {self.get_removed_count()}")
        print(f"Columns: {list(self.df.columns)}")
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            print("\nNumeric columns statistics:")
            for col in numeric_cols[:5]:
                print(f"  {col}: mean={self.df[col].mean():.2f}, std={self.df[col].std():.2f}")