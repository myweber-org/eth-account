
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
        print(f"Removed {removed_count} outliers using IQR method")
        return self
    
    def normalize_minmax(self, columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        for col in columns:
            if col in self.df.columns and pd.api.types.is_numeric_dtype(self.df[col]):
                min_val = self.df[col].min()
                max_val = self.df[col].max()
                if max_val > min_val:
                    self.df[col] = (self.df[col] - min_val) / (max_val - min_val)
        
        print(f"Normalized {len(columns)} columns using Min-Max scaling")
        return self
    
    def fill_missing_median(self, columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        for col in columns:
            if col in self.df.columns and self.df[col].isnull().any():
                median_val = self.df[col].median()
                self.df[col].fillna(median_val, inplace=True)
        
        print(f"Filled missing values with median for {len(columns)} columns")
        return self
    
    def get_cleaned_data(self):
        return self.df
    
    def get_summary(self):
        summary = {
            'original_rows': self.original_shape[0],
            'cleaned_rows': self.df.shape[0],
            'original_columns': self.original_shape[1],
            'cleaned_columns': self.df.shape[1],
            'rows_removed': self.original_shape[0] - self.df.shape[0],
            'missing_values': self.df.isnull().sum().sum()
        }
        return summary

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
    cleaner = DataCleaner(sample_df)
    
    cleaned_df = (cleaner
                 .fill_missing_median()
                 .remove_outliers_iqr(['feature_a', 'feature_b'])
                 .normalize_minmax(['feature_a', 'feature_b', 'feature_c'])
                 .get_cleaned_data())
    
    summary = cleaner.get_summary()
    print(f"Data cleaning summary: {summary}")
    print(f"Cleaned data shape: {cleaned_df.shape}")
    print(f"First 5 rows of cleaned data:\n{cleaned_df.head()}")import pandas as pd
import numpy as np
from pathlib import Path

def clean_csv_data(input_path, output_path=None, strategy='mean'):
    """
    Clean a CSV file by handling missing values.
    
    Args:
        input_path (str): Path to input CSV file
        output_path (str, optional): Path for cleaned CSV. Defaults to None.
        strategy (str): Method for handling missing values: 
                       'mean', 'median', 'mode', or 'drop'
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    try:
        df = pd.read_csv(input_path)
        original_shape = df.shape
        
        if strategy == 'drop':
            df_cleaned = df.dropna()
        elif strategy in ['mean', 'median']:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if strategy == 'mean':
                    fill_value = df[col].mean()
                else:
                    fill_value = df[col].median()
                df[col] = df[col].fillna(fill_value)
            df_cleaned = df
        elif strategy == 'mode':
            for col in df.columns:
                if df[col].dtype == 'object':
                    mode_value = df[col].mode()[0] if not df[col].mode().empty else ''
                    df[col] = df[col].fillna(mode_value)
            df_cleaned = df
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        if output_path:
            df_cleaned.to_csv(output_path, index=False)
            print(f"Cleaned data saved to: {output_path}")
        
        missing_count = df.isnull().sum().sum()
        cleaned_missing = df_cleaned.isnull().sum().sum()
        
        print(f"Original shape: {original_shape}")
        print(f"Missing values before: {missing_count}")
        print(f"Missing values after: {cleaned_missing}")
        print(f"Rows removed: {original_shape[0] - df_cleaned.shape[0]}")
        
        return df_cleaned
        
    except FileNotFoundError:
        print(f"Error: File not found at {input_path}")
        return None
    except pd.errors.EmptyDataError:
        print("Error: The CSV file is empty")
        return None
    except Exception as e:
        print(f"Error during cleaning: {str(e)}")
        return None

def validate_csv_file(file_path):
    """
    Validate if a file exists and is a CSV.
    
    Args:
        file_path (str): Path to the file
    
    Returns:
        bool: True if valid, False otherwise
    """
    path = Path(file_path)
    if not path.exists():
        print(f"File does not exist: {file_path}")
        return False
    if path.suffix.lower() != '.csv':
        print(f"File is not a CSV: {file_path}")
        return False
    return True

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, np.nan, 4, 5],
        'B': [np.nan, 2, 3, np.nan, 5],
        'C': ['x', 'y', np.nan, 'z', 'x']
    }
    
    test_df = pd.DataFrame(sample_data)
    test_df.to_csv('test_data.csv', index=False)
    
    print("Testing data cleaning utility...")
    cleaned = clean_csv_data('test_data.csv', 'cleaned_data.csv', strategy='mean')
    
    import os
    if os.path.exists('test_data.csv'):
        os.remove('test_data.csv')
    if os.path.exists('cleaned_data.csv'):
        os.remove('cleaned_data.csv')