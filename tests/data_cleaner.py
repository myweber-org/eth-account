
import numpy as np
import pandas as pd
from scipy import stats

class DataCleaner:
    def __init__(self, data):
        self.data = data
        self.original_shape = data.shape
        
    def remove_outliers_iqr(self, columns=None, factor=1.5):
        if columns is None:
            columns = self.data.columns
            
        clean_data = self.data.copy()
        for col in columns:
            if pd.api.types.is_numeric_dtype(clean_data[col]):
                Q1 = clean_data[col].quantile(0.25)
                Q3 = clean_data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - factor * IQR
                upper_bound = Q3 + factor * IQR
                clean_data = clean_data[(clean_data[col] >= lower_bound) & (clean_data[col] <= upper_bound)]
        
        self.data = clean_data.reset_index(drop=True)
        return self.data
    
    def normalize_minmax(self, columns=None):
        if columns is None:
            columns = self.data.select_dtypes(include=[np.number]).columns
            
        normalized_data = self.data.copy()
        for col in columns:
            if col in normalized_data.columns and pd.api.types.is_numeric_dtype(normalized_data[col]):
                min_val = normalized_data[col].min()
                max_val = normalized_data[col].max()
                if max_val != min_val:
                    normalized_data[col] = (normalized_data[col] - min_val) / (max_val - min_val)
        
        self.data = normalized_data
        return self.data
    
    def standardize_zscore(self, columns=None):
        if columns is None:
            columns = self.data.select_dtypes(include=[np.number]).columns
            
        standardized_data = self.data.copy()
        for col in columns:
            if col in standardized_data.columns and pd.api.types.is_numeric_dtype(standardized_data[col]):
                mean_val = standardized_data[col].mean()
                std_val = standardized_data[col].std()
                if std_val > 0:
                    standardized_data[col] = (standardized_data[col] - mean_val) / std_val
        
        self.data = standardized_data
        return self.data
    
    def fill_missing_median(self, columns=None):
        if columns is None:
            columns = self.data.select_dtypes(include=[np.number]).columns
            
        filled_data = self.data.copy()
        for col in columns:
            if col in filled_data.columns and pd.api.types.is_numeric_dtype(filled_data[col]):
                filled_data[col] = filled_data[col].fillna(filled_data[col].median())
        
        self.data = filled_data
        return self.data
    
    def get_cleaning_report(self):
        removed_rows = self.original_shape[0] - self.data.shape[0]
        removed_columns = self.original_shape[1] - self.data.shape[1]
        
        report = {
            'original_rows': self.original_shape[0],
            'current_rows': self.data.shape[0],
            'original_columns': self.original_shape[1],
            'current_columns': self.data.shape[1],
            'rows_removed': removed_rows,
            'columns_removed': removed_columns,
            'remaining_missing_values': self.data.isnull().sum().sum()
        }
        
        return report

def create_sample_dataset():
    np.random.seed(42)
    data = {
        'feature_a': np.random.normal(100, 15, 1000),
        'feature_b': np.random.exponential(50, 1000),
        'feature_c': np.random.uniform(0, 1, 1000),
        'category': np.random.choice(['A', 'B', 'C'], 1000)
    }
    
    df = pd.DataFrame(data)
    df.loc[np.random.choice(df.index, 50), 'feature_a'] = np.nan
    df.loc[np.random.choice(df.index, 30), 'feature_b'] = np.nan
    
    outliers = np.random.randint(0, 1000, 20)
    df.loc[outliers, 'feature_a'] = df['feature_a'].mean() + 5 * df['feature_a'].std()
    
    return df

if __name__ == "__main__":
    sample_data = create_sample_dataset()
    cleaner = DataCleaner(sample_data)
    
    print("Initial data shape:", cleaner.data.shape)
    print("Missing values:", cleaner.data.isnull().sum().sum())
    
    cleaner.fill_missing_median()
    cleaner.remove_outliers_iqr(['feature_a', 'feature_b'])
    cleaner.standardize_zscore(['feature_a', 'feature_b', 'feature_c'])
    
    report = cleaner.get_cleaning_report()
    print("\nCleaning Report:")
    for key, value in report.items():
        print(f"{key}: {value}")
import numpy as np
import pandas as pd

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range method.
    
    Args:
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
    
    return filtered_df.reset_index(drop=True)

def calculate_statistics(df, column):
    """
    Calculate basic statistics for a column after outlier removal.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        column (str): Column name to analyze
    
    Returns:
        dict: Dictionary containing statistics
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

def process_dataframe(df, column):
    """
    Complete data processing pipeline: remove outliers and return statistics.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        column (str): Column name to process
    
    Returns:
        tuple: (cleaned_df, statistics_dict)
    """
    cleaned_df = remove_outliers_iqr(df, column)
    stats = calculate_statistics(cleaned_df, column)
    
    return cleaned_df, stats

if __name__ == "__main__":
    sample_data = {
        'values': [10, 12, 12, 13, 12, 11, 14, 13, 15, 102, 12, 14, 13, 12, 11, 14, 13, 12, 14, 105]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print(f"\nOriginal shape: {df.shape}")
    
    cleaned_df, stats = process_dataframe(df, 'values')
    
    print("\nCleaned DataFrame:")
    print(cleaned_df)
    print(f"\nCleaned shape: {cleaned_df.shape}")
    
    print("\nStatistics after cleaning:")
    for key, value in stats.items():
        print(f"{key}: {value:.2f}" if isinstance(value, float) else f"{key}: {value}")