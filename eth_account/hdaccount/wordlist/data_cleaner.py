
import pandas as pd
import numpy as np
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

def clean_dataset(file_path, numeric_columns):
    df = pd.read_csv(file_path)
    
    for col in numeric_columns:
        if col in df.columns:
            df = remove_outliers_iqr(df, col)
            df = normalize_minmax(df, col)
    
    df = df.dropna()
    return df

def calculate_statistics(df, column):
    return {
        'mean': df[column].mean(),
        'median': df[column].median(),
        'std': df[column].std(),
        'skewness': stats.skew(df[column].dropna()),
        'kurtosis': stats.kurtosis(df[column].dropna())
    }

if __name__ == "__main__":
    cleaned_data = clean_dataset('sample_data.csv', ['age', 'income', 'score'])
    print(cleaned_data.head())
    
    stats_result = calculate_statistics(cleaned_data, 'income')
    print(stats_result)
import pandas as pd
import numpy as np
from scipy import stats

def remove_outliers_iqr(dataframe, columns):
    """
    Remove outliers using IQR method for specified columns.
    Returns cleaned dataframe and outlier indices.
    """
    cleaned_df = dataframe.copy()
    outlier_indices = []
    
    for col in columns:
        if col not in cleaned_df.columns:
            continue
            
        Q1 = cleaned_df[col].quantile(0.25)
        Q3 = cleaned_df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        col_outliers = cleaned_df[(cleaned_df[col] < lower_bound) | (cleaned_df[col] > upper_bound)].index
        outlier_indices.extend(col_outliers)
        
        cleaned_df = cleaned_df[(cleaned_df[col] >= lower_bound) & (cleaned_df[col] <= upper_bound)]
    
    return cleaned_df, list(set(outlier_indices))

def normalize_data(dataframe, columns, method='minmax'):
    """
    Normalize specified columns using different methods.
    Supported methods: 'minmax', 'zscore', 'robust'
    """
    normalized_df = dataframe.copy()
    
    for col in columns:
        if col not in normalized_df.columns:
            continue
            
        if method == 'minmax':
            col_min = normalized_df[col].min()
            col_max = normalized_df[col].max()
            if col_max != col_min:
                normalized_df[col] = (normalized_df[col] - col_min) / (col_max - col_min)
            else:
                normalized_df[col] = 0
                
        elif method == 'zscore':
            col_mean = normalized_df[col].mean()
            col_std = normalized_df[col].std()
            if col_std != 0:
                normalized_df[col] = (normalized_df[col] - col_mean) / col_std
            else:
                normalized_df[col] = 0
                
        elif method == 'robust':
            col_median = normalized_df[col].median()
            col_iqr = stats.iqr(normalized_df[col])
            if col_iqr != 0:
                normalized_df[col] = (normalized_df[col] - col_median) / col_iqr
            else:
                normalized_df[col] = 0
    
    return normalized_df

def handle_missing_values(dataframe, strategy='mean', fill_value=None):
    """
    Handle missing values in numeric columns.
    Strategies: 'mean', 'median', 'mode', 'constant', 'drop'
    """
    df_copy = dataframe.copy()
    numeric_cols = df_copy.select_dtypes(include=[np.number]).columns
    
    if strategy == 'drop':
        return df_copy.dropna(subset=numeric_cols)
    
    for col in numeric_cols:
        if df_copy[col].isnull().any():
            if strategy == 'mean':
                fill_val = df_copy[col].mean()
            elif strategy == 'median':
                fill_val = df_copy[col].median()
            elif strategy == 'mode':
                fill_val = df_copy[col].mode()[0] if not df_copy[col].mode().empty else 0
            elif strategy == 'constant':
                fill_val = fill_value if fill_value is not None else 0
            else:
                fill_val = 0
                
            df_copy[col] = df_copy[col].fillna(fill_val)
    
    return df_copy

def clean_dataset(dataframe, numeric_columns=None, outlier_method='iqr', 
                  normalize_method='minmax', missing_strategy='mean'):
    """
    Main function to clean dataset with multiple steps.
    """
    if numeric_columns is None:
        numeric_columns = dataframe.select_dtypes(include=[np.number]).columns.tolist()
    
    print(f"Original dataset shape: {dataframe.shape}")
    
    # Handle missing values
    cleaned_data = handle_missing_values(dataframe, strategy=missing_strategy)
    print(f"After handling missing values: {cleaned_data.shape}")
    
    # Remove outliers
    if outlier_method == 'iqr':
        cleaned_data, outliers = remove_outliers_iqr(cleaned_data, numeric_columns)
        print(f"Removed {len(outliers)} outlier rows")
        print(f"After outlier removal: {cleaned_data.shape}")
    
    # Normalize data
    normalized_data = normalize_data(cleaned_data, numeric_columns, method=normalize_method)
    
    return normalized_data

if __name__ == "__main__":
    # Example usage
    sample_data = pd.DataFrame({
        'feature1': np.random.normal(100, 15, 1000),
        'feature2': np.random.exponential(50, 1000),
        'feature3': np.random.uniform(0, 1, 1000),
        'category': np.random.choice(['A', 'B', 'C'], 1000)
    })
    
    # Add some missing values
    sample_data.loc[np.random.choice(sample_data.index, 50), 'feature1'] = np.nan
    
    # Clean the dataset
    cleaned = clean_dataset(
        sample_data,
        numeric_columns=['feature1', 'feature2', 'feature3'],
        outlier_method='iqr',
        normalize_method='minmax',
        missing_strategy='mean'
    )
    
    print(f"Final cleaned dataset shape: {cleaned.shape}")
    print(f"Cleaned dataset statistics:\n{cleaned.describe()}")
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
            if col in df_clean.columns:
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
        
        removed_count = len(self.df) - len(df_clean)
        self.df = df_clean
        return removed_count
    
    def normalize_data(self, method='minmax', columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
            
        df_normalized = self.df.copy()
        
        if method == 'minmax':
            for col in columns:
                if col in df_normalized.columns:
                    min_val = df_normalized[col].min()
                    max_val = df_normalized[col].max()
                    if max_val > min_val:
                        df_normalized[col] = (df_normalized[col] - min_val) / (max_val - min_val)
        
        elif method == 'zscore':
            for col in columns:
                if col in df_normalized.columns:
                    mean_val = df_normalized[col].mean()
                    std_val = df_normalized[col].std()
                    if std_val > 0:
                        df_normalized[col] = (df_normalized[col] - mean_val) / std_val
        
        self.df = df_normalized
        return self.df
    
    def handle_missing_values(self, strategy='mean', columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
            
        df_filled = self.df.copy()
        
        for col in columns:
            if col in df_filled.columns and df_filled[col].isnull().any():
                if strategy == 'mean':
                    fill_value = df_filled[col].mean()
                elif strategy == 'median':
                    fill_value = df_filled[col].median()
                elif strategy == 'mode':
                    fill_value = df_filled[col].mode()[0]
                else:
                    fill_value = 0
                
                df_filled[col] = df_filled[col].fillna(fill_value)
        
        self.df = df_filled
        return self.df
    
    def get_summary(self):
        summary = {
            'original_rows': self.original_shape[0],
            'current_rows': len(self.df),
            'original_columns': self.original_shape[1],
            'current_columns': len(self.df.columns),
            'rows_removed': self.original_shape[0] - len(self.df),
            'missing_values': self.df.isnull().sum().sum()
        }
        return summary
    
    def get_cleaned_data(self):
        return self.df.copy()

def create_sample_data():
    np.random.seed(42)
    data = {
        'feature_a': np.random.normal(100, 15, 100),
        'feature_b': np.random.exponential(50, 100),
        'feature_c': np.random.uniform(0, 1, 100),
        'category': np.random.choice(['A', 'B', 'C'], 100)
    }
    
    df = pd.DataFrame(data)
    df.loc[np.random.choice(df.index, 5), 'feature_a'] = np.nan
    df.loc[np.random.choice(df.index, 3), 'feature_b'] = np.nan
    
    outliers = np.random.randint(0, 100, 10)
    df.loc[outliers, 'feature_a'] = df['feature_a'].mean() + 5 * df['feature_a'].std()
    
    return df

if __name__ == "__main__":
    sample_df = create_sample_data()
    cleaner = DataCleaner(sample_df)
    
    print("Initial data shape:", cleaner.df.shape)
    print("Missing values:", cleaner.df.isnull().sum().sum())
    
    removed = cleaner.remove_outliers_iqr(['feature_a', 'feature_b'])
    print(f"Removed {removed} outliers")
    
    cleaner.handle_missing_values(strategy='mean')
    print("Missing values after handling:", cleaner.df.isnull().sum().sum())
    
    cleaner.normalize_data(method='minmax')
    
    summary = cleaner.get_summary()
    print("\nCleaning Summary:")
    for key, value in summary.items():
        print(f"{key}: {value}")
    
    cleaned_df = cleaner.get_cleaned_data()
    print(f"\nFinal data shape: {cleaned_df.shape}")
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
    
    return filtered_df.reset_index(drop=True)

def calculate_basic_stats(df, column):
    """
    Calculate basic statistics for a DataFrame column.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to analyze
    
    Returns:
    dict: Dictionary containing statistical measures
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    stats = {
        'mean': df[column].mean(),
        'median': df[column].median(),
        'std': df[column].std(),
        'min': df[column].min(),
        'max': df[column].max(),
        'count': df[column].count(),
        'missing': df[column].isnull().sum()
    }
    
    return stats

def normalize_column(df, column, method='minmax'):
    """
    Normalize a DataFrame column using specified method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to normalize
    method (str): Normalization method ('minmax' or 'zscore')
    
    Returns:
    pd.DataFrame: DataFrame with normalized column
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    df_copy = df.copy()
    
    if method == 'minmax':
        min_val = df_copy[column].min()
        max_val = df_copy[column].max()
        if max_val != min_val:
            df_copy[f'{column}_normalized'] = (df_copy[column] - min_val) / (max_val - min_val)
        else:
            df_copy[f'{column}_normalized'] = 0
    
    elif method == 'zscore':
        mean_val = df_copy[column].mean()
        std_val = df_copy[column].std()
        if std_val != 0:
            df_copy[f'{column}_normalized'] = (df_copy[column] - mean_val) / std_val
        else:
            df_copy[f'{column}_normalized'] = 0
    
    else:
        raise ValueError("Method must be 'minmax' or 'zscore'")
    
    return df_copy

if __name__ == "__main__":
    sample_data = {
        'values': [10, 12, 12, 13, 12, 11, 14, 13, 15, 100, 12, 13, 12, 11, 10, 14, 13, 12, 11, 10]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original data:")
    print(df)
    print("\nBasic statistics:")
    print(calculate_basic_stats(df, 'values'))
    
    cleaned_df = remove_outliers_iqr(df, 'values')
    print("\nData after outlier removal:")
    print(cleaned_df)
    print("\nStatistics after cleaning:")
    print(calculate_basic_stats(cleaned_df, 'values'))
    
    normalized_df = normalize_column(cleaned_df, 'values', method='minmax')
    print("\nData after min-max normalization:")
    print(normalized_df)