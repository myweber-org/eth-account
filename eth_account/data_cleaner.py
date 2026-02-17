
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to process
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed
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

def calculate_summary_statistics(df, column):
    """
    Calculate summary statistics for a column after outlier removal.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to analyze
    
    Returns:
    dict: Dictionary containing summary statistics
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    stats = {
        'mean': df[column].mean(),
        'median': df[column].median(),
        'std': df[column].std(),
        'min': df[column].min(),
        'max': df[column].max(),
        'count': df[column].count()
    }
    
    return stats

def clean_dataset(df, numeric_columns=None):
    """
    Clean dataset by removing outliers from all numeric columns.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    numeric_columns (list): List of numeric column names to clean
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_df = df.copy()
    
    for column in numeric_columns:
        if column in df.columns and pd.api.types.is_numeric_dtype(df[column]):
            original_count = len(cleaned_df)
            cleaned_df = remove_outliers_iqr(cleaned_df, column)
            removed_count = original_count - len(cleaned_df)
            print(f"Removed {removed_count} outliers from column '{column}'")
    
    return cleaned_df

if __name__ == "__main__":
    sample_data = {
        'A': np.random.normal(100, 15, 1000),
        'B': np.random.exponential(50, 1000),
        'C': np.random.uniform(0, 200, 1000)
    }
    
    df = pd.DataFrame(sample_data)
    df.loc[::100, 'A'] = 500
    
    print("Original dataset shape:", df.shape)
    print("\nOriginal statistics for column 'A':")
    print(calculate_summary_statistics(df, 'A'))
    
    cleaned_df = clean_dataset(df, ['A', 'B'])
    
    print("\nCleaned dataset shape:", cleaned_df.shape)
    print("\nCleaned statistics for column 'A':")
    print(calculate_summary_statistics(cleaned_df, 'A'))
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
        
        clean_df = self.df.copy()
        for col in columns:
            if col in clean_df.columns:
                Q1 = clean_df[col].quantile(0.25)
                Q3 = clean_df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                clean_df = clean_df[(clean_df[col] >= lower_bound) & (clean_df[col] <= upper_bound)]
        
        removed_count = len(self.df) - len(clean_df)
        self.df = clean_df
        return removed_count
    
    def normalize_data(self, columns=None, method='zscore'):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        normalized_df = self.df.copy()
        
        if method == 'zscore':
            for col in columns:
                if col in normalized_df.columns:
                    normalized_df[col] = stats.zscore(normalized_df[col])
        
        elif method == 'minmax':
            for col in columns:
                if col in normalized_df.columns:
                    col_min = normalized_df[col].min()
                    col_max = normalized_df[col].max()
                    if col_max != col_min:
                        normalized_df[col] = (normalized_df[col] - col_min) / (col_max - col_min)
        
        self.df = normalized_df
        return self.df
    
    def handle_missing_values(self, strategy='mean', columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        filled_df = self.df.copy()
        
        for col in columns:
            if col in filled_df.columns and filled_df[col].isnull().any():
                if strategy == 'mean':
                    fill_value = filled_df[col].mean()
                elif strategy == 'median':
                    fill_value = filled_df[col].median()
                elif strategy == 'mode':
                    fill_value = filled_df[col].mode()[0]
                else:
                    fill_value = 0
                
                filled_df[col] = filled_df[col].fillna(fill_value)
        
        self.df = filled_df
        return self.df
    
    def get_cleaned_data(self):
        return self.df
    
    def get_cleaning_stats(self):
        stats_dict = {
            'original_rows': self.original_shape[0],
            'cleaned_rows': len(self.df),
            'rows_removed': self.original_shape[0] - len(self.df),
            'removal_percentage': ((self.original_shape[0] - len(self.df)) / self.original_shape[0]) * 100
        }
        return stats_dict

def create_sample_data():
    np.random.seed(42)
    data = {
        'feature_a': np.random.normal(100, 15, 1000),
        'feature_b': np.random.exponential(50, 1000),
        'feature_c': np.random.uniform(0, 200, 1000),
        'category': np.random.choice(['A', 'B', 'C'], 1000)
    }
    
    df = pd.DataFrame(data)
    
    df.loc[np.random.choice(df.index, 50), 'feature_a'] = np.nan
    df.loc[np.random.choice(df.index, 30), 'feature_b'] = np.nan
    
    outlier_indices = np.random.choice(df.index, 20)
    df.loc[outlier_indices, 'feature_c'] = df['feature_c'].max() * 5
    
    return df

if __name__ == "__main__":
    sample_df = create_sample_data()
    print("Sample data created with shape:", sample_df.shape)
    
    cleaner = DataCleaner(sample_df)
    
    print("\nHandling missing values...")
    cleaner.handle_missing_values(strategy='mean')
    
    print("\nRemoving outliers...")
    outliers_removed = cleaner.remove_outliers_iqr(threshold=1.5)
    print(f"Removed {outliers_removed} outliers")
    
    print("\nNormalizing data...")
    cleaner.normalize_data(method='zscore')
    
    cleaned_df = cleaner.get_cleaned_data()
    stats = cleaner.get_cleaning_stats()
    
    print(f"\nCleaning Statistics:")
    print(f"Original rows: {stats['original_rows']}")
    print(f"Cleaned rows: {stats['cleaned_rows']}")
    print(f"Rows removed: {stats['rows_removed']}")
    print(f"Removal percentage: {stats['removal_percentage']:.2f}%")
    
    print(f"\nCleaned data shape: {cleaned_df.shape}")
    print("\nFirst few rows of cleaned data:")
    print(cleaned_df.head())