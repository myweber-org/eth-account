
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
        return self
    
    def remove_outliers_zscore(self, columns=None, threshold=3):
        if columns is None:
            columns = self.data.columns
            
        clean_data = self.data.copy()
        for col in columns:
            if pd.api.types.is_numeric_dtype(clean_data[col]):
                z_scores = np.abs(stats.zscore(clean_data[col].dropna()))
                clean_data = clean_data[(z_scores < threshold) | clean_data[col].isna()]
        
        self.data = clean_data.reset_index(drop=True)
        return self
    
    def normalize_minmax(self, columns=None):
        if columns is None:
            columns = self.data.select_dtypes(include=[np.number]).columns
            
        normalized_data = self.data.copy()
        for col in columns:
            if pd.api.types.is_numeric_dtype(normalized_data[col]):
                col_min = normalized_data[col].min()
                col_max = normalized_data[col].max()
                if col_max != col_min:
                    normalized_data[col] = (normalized_data[col] - col_min) / (col_max - col_min)
        
        self.data = normalized_data
        return self
    
    def normalize_zscore(self, columns=None):
        if columns is None:
            columns = self.data.select_dtypes(include=[np.number]).columns
            
        normalized_data = self.data.copy()
        for col in columns:
            if pd.api.types.is_numeric_dtype(normalized_data[col]):
                col_mean = normalized_data[col].mean()
                col_std = normalized_data[col].std()
                if col_std > 0:
                    normalized_data[col] = (normalized_data[col] - col_mean) / col_std
        
        self.data = normalized_data
        return self
    
    def fill_missing_mean(self, columns=None):
        if columns is None:
            columns = self.data.select_dtypes(include=[np.number]).columns
            
        filled_data = self.data.copy()
        for col in columns:
            if pd.api.types.is_numeric_dtype(filled_data[col]):
                filled_data[col] = filled_data[col].fillna(filled_data[col].mean())
        
        self.data = filled_data
        return self
    
    def fill_missing_median(self, columns=None):
        if columns is None:
            columns = self.data.select_dtypes(include=[np.number]).columns
            
        filled_data = self.data.copy()
        for col in columns:
            if pd.api.types.is_numeric_dtype(filled_data[col]):
                filled_data[col] = filled_data[col].fillna(filled_data[col].median())
        
        self.data = filled_data
        return self
    
    def get_cleaned_data(self):
        return self.data
    
    def get_removed_count(self):
        return self.original_shape[0] - self.data.shape[0]
    
    def get_summary(self):
        summary = {
            'original_rows': self.original_shape[0],
            'cleaned_rows': self.data.shape[0],
            'removed_rows': self.get_removed_count(),
            'columns': list(self.data.columns),
            'missing_values': self.data.isnull().sum().to_dict(),
            'data_types': self.data.dtypes.to_dict()
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
import numpy as np

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a pandas DataFrame column using the IQR method.
    
    Parameters:
    data (pd.DataFrame): The input DataFrame.
    column (str): The column name to process.
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed from the specified column.
    """
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data

def calculate_summary_statistics(data, column):
    """
    Calculate summary statistics for a column after outlier removal.
    
    Parameters:
    data (pd.DataFrame): The input DataFrame.
    column (str): The column name to analyze.
    
    Returns:
    dict: Dictionary containing count, mean, std, min, and max.
    """
    stats = {
        'count': data[column].count(),
        'mean': data[column].mean(),
        'std': data[column].std(),
        'min': data[column].min(),
        'max': data[column].max()
    }
    return stats

if __name__ == "__main__":
    import pandas as pd
    
    sample_data = pd.DataFrame({
        'values': [10, 12, 12, 13, 14, 15, 15, 16, 17, 18, 100]
    })
    
    print("Original data:")
    print(sample_data)
    
    cleaned_data = remove_outliers_iqr(sample_data, 'values')
    print("\nData after outlier removal:")
    print(cleaned_data)
    
    stats = calculate_summary_statistics(cleaned_data, 'values')
    print("\nSummary statistics:")
    for key, value in stats.items():
        print(f"{key}: {value:.2f}")