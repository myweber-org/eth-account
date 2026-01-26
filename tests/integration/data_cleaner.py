
import pandas as pd
import numpy as np
from scipy import stats

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.original_shape = df.shape
        
    def remove_outliers_iqr(self, columns=None, factor=1.5):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        clean_df = self.df.copy()
        for col in columns:
            if col in self.df.columns and self.df[col].dtype in [np.float64, np.int64]:
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - factor * IQR
                upper_bound = Q3 + factor * IQR
                
                mask = (self.df[col] >= lower_bound) & (self.df[col] <= upper_bound)
                clean_df = clean_df[mask]
        
        self.df = clean_df.reset_index(drop=True)
        return self
    
    def remove_outliers_zscore(self, columns=None, threshold=3):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        clean_df = self.df.copy()
        for col in columns:
            if col in self.df.columns and self.df[col].dtype in [np.float64, np.int64]:
                z_scores = np.abs(stats.zscore(self.df[col].fillna(self.df[col].mean())))
                mask = z_scores < threshold
                clean_df = clean_df[mask]
        
        self.df = clean_df.reset_index(drop=True)
        return self
    
    def normalize_minmax(self, columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        normalized_df = self.df.copy()
        for col in columns:
            if col in self.df.columns and self.df[col].dtype in [np.float64, np.int64]:
                min_val = self.df[col].min()
                max_val = self.df[col].max()
                if max_val != min_val:
                    normalized_df[col] = (self.df[col] - min_val) / (max_val - min_val)
        
        self.df = normalized_df
        return self
    
    def normalize_zscore(self, columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        normalized_df = self.df.copy()
        for col in columns:
            if col in self.df.columns and self.df[col].dtype in [np.float64, np.int64]:
                mean_val = self.df[col].mean()
                std_val = self.df[col].std()
                if std_val > 0:
                    normalized_df[col] = (self.df[col] - mean_val) / std_val
        
        self.df = normalized_df
        return self
    
    def fill_missing_mean(self, columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        filled_df = self.df.copy()
        for col in columns:
            if col in self.df.columns and self.df[col].dtype in [np.float64, np.int64]:
                filled_df[col] = self.df[col].fillna(self.df[col].mean())
        
        self.df = filled_df
        return self
    
    def fill_missing_median(self, columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        filled_df = self.df.copy()
        for col in columns:
            if col in self.df.columns and self.df[col].dtype in [np.float64, np.int64]:
                filled_df[col] = self.df[col].fillna(self.df[col].median())
        
        self.df = filled_df
        return self
    
    def get_cleaned_data(self):
        return self.df
    
    def get_removed_count(self):
        return self.original_shape[0] - self.df.shape[0]
    
    def get_summary(self):
        summary = {
            'original_rows': self.original_shape[0],
            'cleaned_rows': self.df.shape[0],
            'removed_rows': self.get_removed_count(),
            'columns': self.df.shape[1],
            'missing_values': self.df.isnull().sum().sum(),
            'numeric_columns': list(self.df.select_dtypes(include=[np.number]).columns)
        }
        return summary

def clean_dataset(df, method='iqr', normalize=False, fill_missing=True):
    cleaner = DataCleaner(df)
    
    if method == 'iqr':
        cleaner.remove_outliers_iqr()
    elif method == 'zscore':
        cleaner.remove_outliers_zscore()
    
    if fill_missing:
        cleaner.fill_missing_mean()
    
    if normalize:
        cleaner.normalize_minmax()
    
    return cleaner.get_cleaned_data(), cleaner.get_summary()
import pandas as pd

def clean_dataset(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    drop_duplicates (bool): Whether to drop duplicate rows. Default is True.
    fill_missing (str): Method to fill missing values. Options: 'mean', 'median', 'mode', or 'drop'. Default is 'mean'.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    if fill_missing == 'drop':
        cleaned_df = cleaned_df.dropna()
    elif fill_missing in ['mean', 'median']:
        numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if fill_missing == 'mean':
                cleaned_df[col].fillna(cleaned_df[col].mean(), inplace=True)
            elif fill_missing == 'median':
                cleaned_df[col].fillna(cleaned_df[col].median(), inplace=True)
    elif fill_missing == 'mode':
        for col in cleaned_df.columns:
            cleaned_df[col].fillna(cleaned_df[col].mode()[0] if not cleaned_df[col].mode().empty else None, inplace=True)
    
    return cleaned_df

def validate_data(df, required_columns=None, min_rows=1):
    """
    Validate the DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of column names that must be present.
    min_rows (int): Minimum number of rows required.
    
    Returns:
    bool: True if validation passes, False otherwise.
    """
    if len(df) < min_rows:
        return False
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False
    
    return True

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, 2, None, 5],
        'B': [10, None, 30, 40, 50],
        'C': ['x', 'y', 'y', 'z', None]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    
    cleaned = clean_dataset(df, fill_missing='mean')
    print("\nCleaned DataFrame:")
    print(cleaned)
    
    is_valid = validate_data(cleaned, required_columns=['A', 'B'], min_rows=3)
    print(f"\nData validation passed: {is_valid}")import pandas as pd
import numpy as np
from scipy import stats

def clean_dataset(df, numeric_columns=None, z_threshold=3, normalize=True):
    """
    Clean dataset by handling missing values, removing outliers, and normalizing.
    
    Parameters:
    df: pandas DataFrame
    numeric_columns: list of column names to process (default: all numeric columns)
    z_threshold: z-score threshold for outlier removal
    normalize: whether to apply min-max normalization
    
    Returns:
    Cleaned pandas DataFrame
    """
    
    df_clean = df.copy()
    
    if numeric_columns is None:
        numeric_columns = df_clean.select_dtypes(include=[np.number]).columns.tolist()
    
    for col in numeric_columns:
        if col not in df_clean.columns:
            continue
            
        series = df_clean[col]
        
        if series.dtype in [np.float64, np.int64]:
            median_val = series.median()
            df_clean[col] = series.fillna(median_val)
            
            z_scores = np.abs(stats.zscore(df_clean[col]))
            df_clean = df_clean[z_scores < z_threshold]
            
            if normalize and len(df_clean[col]) > 0:
                min_val = df_clean[col].min()
                max_val = df_clean[col].max()
                if max_val > min_val:
                    df_clean[col] = (df_clean[col] - min_val) / (max_val - min_val)
    
    df_clean = df_clean.reset_index(drop=True)
    return df_clean

def detect_outliers_iqr(df, column):
    """
    Detect outliers using IQR method.
    
    Parameters:
    df: pandas DataFrame
    column: column name to check
    
    Returns:
    Tuple of (lower_bound, upper_bound, outlier_indices)
    """
    if column not in df.columns:
        return None, None, []
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    outlier_indices = outliers.index.tolist()
    
    return lower_bound, upper_bound, outlier_indices

def validate_dataframe(df, required_columns=None):
    """
    Validate dataframe structure and content.
    
    Parameters:
    df: pandas DataFrame to validate
    required_columns: list of required column names
    
    Returns:
    Tuple of (is_valid, error_message)
    """
    if not isinstance(df, pd.DataFrame):
        return False, "Input is not a pandas DataFrame"
    
    if df.empty:
        return False, "DataFrame is empty"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    return True, "DataFrame is valid"

if __name__ == "__main__":
    sample_data = {
        'feature1': [1, 2, 3, 4, 5, 100],
        'feature2': [10, 20, 30, 40, 50, 60],
        'feature3': [100, 200, 300, 400, 500, 600]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nDataFrame info:")
    print(df.info())
    
    cleaned_df = clean_dataset(df, z_threshold=2)
    print("\nCleaned DataFrame:")
    print(cleaned_df)
    
    is_valid, message = validate_dataframe(cleaned_df)
    print(f"\nValidation: {is_valid} - {message}")
    
    lower, upper, outliers = detect_outliers_iqr(df, 'feature1')
    print(f"\nOutlier detection for 'feature1':")
    print(f"Lower bound: {lower}, Upper bound: {upper}")
    print(f"Outlier indices: {outliers}")