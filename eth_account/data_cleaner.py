
import numpy as np
import pandas as pd
from scipy import stats

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.numeric_columns = df.select_dtypes(include=[np.number]).columns
        
    def remove_outliers_iqr(self, column, multiplier=1.5):
        if column not in self.numeric_columns:
            return self.df
        
        Q1 = self.df[column].quantile(0.25)
        Q3 = self.df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        
        return self.df[(self.df[column] >= lower_bound) & (self.df[column] <= upper_bound)]
    
    def remove_outliers_zscore(self, column, threshold=3):
        if column not in self.numeric_columns:
            return self.df
        
        z_scores = np.abs(stats.zscore(self.df[column]))
        return self.df[z_scores < threshold]
    
    def normalize_minmax(self, column):
        if column not in self.numeric_columns:
            return self.df
        
        min_val = self.df[column].min()
        max_val = self.df[column].max()
        self.df[column + '_normalized'] = (self.df[column] - min_val) / (max_val - min_val)
        return self.df
    
    def standardize_zscore(self, column):
        if column not in self.numeric_columns:
            return self.df
        
        mean_val = self.df[column].mean()
        std_val = self.df[column].std()
        self.df[column + '_standardized'] = (self.df[column] - mean_val) / std_val
        return self.df
    
    def handle_missing_mean(self, column):
        if column not in self.numeric_columns:
            return self.df
        
        mean_val = self.df[column].mean()
        self.df[column].fillna(mean_val, inplace=True)
        return self.df
    
    def get_summary(self):
        summary = {
            'total_rows': len(self.df),
            'total_columns': len(self.df.columns),
            'numeric_columns': list(self.numeric_columns),
            'missing_values': self.df.isnull().sum().to_dict(),
            'data_types': self.df.dtypes.to_dict()
        }
        return summary
import numpy as np
import pandas as pd

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to clean
    
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
        'count': len(df[column])
    }
    
    return stats

def process_dataset(file_path, column_to_clean):
    """
    Main function to load, clean, and analyze a dataset.
    
    Parameters:
    file_path (str): Path to CSV file
    column_to_clean (str): Column name to clean
    
    Returns:
    tuple: (cleaned_df, original_stats, cleaned_stats)
    """
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    
    original_stats = calculate_summary_statistics(df, column_to_clean)
    cleaned_df = remove_outliers_iqr(df, column_to_clean)
    cleaned_stats = calculate_summary_statistics(cleaned_df, column_to_clean)
    
    return cleaned_df, original_stats, cleaned_stats

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'values': np.concatenate([
            np.random.normal(100, 10, 90),
            np.random.normal(300, 50, 10)
        ])
    })
    
    cleaned_data, original_stats, cleaned_stats = process_dataset(
        None, 
        'values'
    )
    
    print("Original data shape:", sample_data.shape)
    print("Cleaned data shape:", cleaned_data.shape)
    print("\nOriginal statistics:", original_stats)
    print("\nCleaned statistics:", cleaned_stats)