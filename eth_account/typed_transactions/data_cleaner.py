import numpy as np
import pandas as pd

def remove_outliers_iqr(df, columns, factor=1.5):
    """
    Remove outliers using IQR method.
    
    Parameters:
    df (pd.DataFrame): Input dataframe
    columns (list): List of column names to process
    factor (float): IQR multiplier for outlier detection
    
    Returns:
    pd.DataFrame: Dataframe with outliers removed
    """
    df_clean = df.copy()
    
    for col in columns:
        if col not in df_clean.columns:
            continue
            
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR
        
        mask = (df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)
        df_clean = df_clean[mask]
    
    return df_clean.reset_index(drop=True)

def normalize_minmax(df, columns):
    """
    Normalize columns using min-max scaling.
    
    Parameters:
    df (pd.DataFrame): Input dataframe
    columns (list): List of column names to normalize
    
    Returns:
    pd.DataFrame: Dataframe with normalized columns
    """
    df_norm = df.copy()
    
    for col in columns:
        if col not in df_norm.columns:
            continue
            
        min_val = df_norm[col].min()
        max_val = df_norm[col].max()
        
        if max_val != min_val:
            df_norm[col] = (df_norm[col] - min_val) / (max_val - min_val)
        else:
            df_norm[col] = 0
    
    return df_norm

def handle_missing_values(df, strategy='mean', columns=None):
    """
    Handle missing values in dataframe.
    
    Parameters:
    df (pd.DataFrame): Input dataframe
    strategy (str): Strategy for imputation ('mean', 'median', 'mode', 'drop')
    columns (list): List of columns to process, None for all numeric columns
    
    Returns:
    pd.DataFrame: Dataframe with handled missing values
    """
    df_processed = df.copy()
    
    if columns is None:
        columns = df_processed.select_dtypes(include=[np.number]).columns.tolist()
    
    if strategy == 'drop':
        df_processed = df_processed.dropna(subset=columns)
    else:
        for col in columns:
            if col not in df_processed.columns:
                continue
                
            if df_processed[col].isnull().any():
                if strategy == 'mean':
                    fill_value = df_processed[col].mean()
                elif strategy == 'median':
                    fill_value = df_processed[col].median()
                elif strategy == 'mode':
                    fill_value = df_processed[col].mode()[0]
                else:
                    fill_value = 0
                
                df_processed[col] = df_processed[col].fillna(fill_value)
    
    return df_processed.reset_index(drop=True)

def clean_dataset(df, numeric_columns, outlier_factor=1.5, normalize=True, missing_strategy='mean'):
    """
    Complete data cleaning pipeline.
    
    Parameters:
    df (pd.DataFrame): Input dataframe
    numeric_columns (list): List of numeric column names to process
    outlier_factor (float): IQR multiplier for outlier detection
    normalize (bool): Whether to apply min-max normalization
    missing_strategy (str): Strategy for handling missing values
    
    Returns:
    pd.DataFrame: Cleaned dataframe
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame")
    
    if not numeric_columns:
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    df_clean = handle_missing_values(df, strategy=missing_strategy, columns=numeric_columns)
    df_clean = remove_outliers_iqr(df_clean, numeric_columns, factor=outlier_factor)
    
    if normalize:
        df_clean = normalize_minmax(df_clean, numeric_columns)
    
    return df_clean
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
            if col in df_clean.columns and pd.api.types.is_numeric_dtype(df_clean[col]):
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
        
        removed_count = self.original_shape[0] - df_clean.shape[0]
        self.df = df_clean
        return removed_count
    
    def normalize_minmax(self, columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
            
        df_normalized = self.df.copy()
        for col in columns:
            if col in df_normalized.columns and pd.api.types.is_numeric_dtype(df_normalized[col]):
                min_val = df_normalized[col].min()
                max_val = df_normalized[col].max()
                if max_val > min_val:
                    df_normalized[col] = (df_normalized[col] - min_val) / (max_val - min_val)
        
        self.df = df_normalized
        return self
    
    def standardize_zscore(self, columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
            
        df_standardized = self.df.copy()
        for col in columns:
            if col in df_standardized.columns and pd.api.types.is_numeric_dtype(df_standardized[col]):
                mean_val = df_standardized[col].mean()
                std_val = df_standardized[col].std()
                if std_val > 0:
                    df_standardized[col] = (df_standardized[col] - mean_val) / std_val
        
        self.df = df_standardized
        return self
    
    def handle_missing_values(self, strategy='mean', columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
            
        df_filled = self.df.copy()
        for col in columns:
            if col in df_filled.columns and pd.api.types.is_numeric_dtype(df_filled[col]):
                if strategy == 'mean':
                    fill_value = df_filled[col].mean()
                elif strategy == 'median':
                    fill_value = df_filled[col].median()
                elif strategy == 'mode':
                    fill_value = df_filled[col].mode()[0] if not df_filled[col].mode().empty else 0
                else:
                    fill_value = 0
                
                df_filled[col] = df_filled[col].fillna(fill_value)
        
        self.df = df_filled
        return self
    
    def get_cleaned_data(self):
        return self.df.copy()
    
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
    
    indices = np.random.choice(df.index, size=50, replace=False)
    for idx in indices:
        df.loc[idx, 'feature_a'] = np.random.normal(300, 50)
    
    nan_indices = np.random.choice(df.index, size=30, replace=False)
    df.loc[nan_indices, 'feature_b'] = np.nan
    
    return df

if __name__ == "__main__":
    sample_df = create_sample_data()
    print("Original data shape:", sample_df.shape)
    print("Missing values:", sample_df.isnull().sum().sum())
    
    cleaner = DataCleaner(sample_df)
    
    outliers_removed = cleaner.remove_outliers_iqr(['feature_a', 'feature_b'])
    print(f"Removed {outliers_removed} outliers")
    
    cleaner.handle_missing_values(strategy='median')
    cleaner.normalize_minmax(['feature_a', 'feature_b', 'feature_c'])
    
    cleaned_df = cleaner.get_cleaned_data()
    summary = cleaner.get_summary()
    
    print("\nCleaning Summary:")
    for key, value in summary.items():
        print(f"{key}: {value}")
    
    print("\nCleaned data shape:", cleaned_df.shape)
    print("Cleaned data statistics:")
    print(cleaned_df.describe())