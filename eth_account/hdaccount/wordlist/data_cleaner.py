
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column, factor=1.5):
    """
    Remove outliers using IQR method
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - factor * IQR
    upper_bound = Q3 + factor * IQR
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    removed_count = len(data) - len(filtered_data)
    
    return filtered_data, removed_count

def normalize_minmax(data, column):
    """
    Normalize data using min-max scaling
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    min_val = data[column].min()
    max_val = data[column].max()
    
    if max_val == min_val:
        return data[column].apply(lambda x: 0.5)
    
    normalized = (data[column] - min_val) / (max_val - min_val)
    return normalized

def standardize_zscore(data, column):
    """
    Standardize data using z-score normalization
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    mean_val = data[column].mean()
    std_val = data[column].std()
    
    if std_val == 0:
        return data[column].apply(lambda x: 0)
    
    standardized = (data[column] - mean_val) / std_val
    return standardized

def handle_missing_values(data, strategy='mean', columns=None):
    """
    Handle missing values with different strategies
    """
    if columns is None:
        columns = data.columns
    
    data_clean = data.copy()
    
    for column in columns:
        if column not in data.columns:
            continue
            
        if data[column].isnull().any():
            if strategy == 'mean':
                fill_value = data[column].mean()
            elif strategy == 'median':
                fill_value = data[column].median()
            elif strategy == 'mode':
                fill_value = data[column].mode()[0]
            elif strategy == 'drop':
                data_clean = data_clean.dropna(subset=[column])
                continue
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
            
            data_clean[column] = data_clean[column].fillna(fill_value)
    
    return data_clean

def create_sample_data():
    """
    Create sample data for testing
    """
    np.random.seed(42)
    data = {
        'feature_a': np.random.normal(100, 15, 100),
        'feature_b': np.random.exponential(50, 100),
        'feature_c': np.random.uniform(0, 1, 100)
    }
    
    df = pd.DataFrame(data)
    
    df.loc[np.random.choice(df.index, size=5), 'feature_a'] = np.nan
    df.loc[np.random.choice(df.index, size=3), 'feature_b'] = np.nan
    
    return df

def main():
    """
    Example usage of data cleaning functions
    """
    print("Creating sample data...")
    df = create_sample_data()
    print(f"Original data shape: {df.shape}")
    print(f"Missing values:\n{df.isnull().sum()}")
    
    print("\nHandling missing values...")
    df_clean = handle_missing_values(df, strategy='mean')
    print(f"After cleaning shape: {df_clean.shape}")
    
    print("\nRemoving outliers from feature_a...")
    df_filtered, removed = remove_outliers_iqr(df_clean, 'feature_a')
    print(f"Removed {removed} outliers")
    print(f"Filtered data shape: {df_filtered.shape}")
    
    print("\nNormalizing feature_b...")
    df_filtered['feature_b_normalized'] = normalize_minmax(df_filtered, 'feature_b')
    print("Normalization complete")
    
    print("\nStandardizing feature_a...")
    df_filtered['feature_a_standardized'] = standardize_zscore(df_filtered, 'feature_a')
    print("Standardization complete")
    
    print("\nSummary statistics:")
    print(df_filtered.describe())
    
    return df_filtered

if __name__ == "__main__":
    cleaned_data = main()
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

def calculate_summary_stats(df, column):
    """
    Calculate summary statistics for a DataFrame column.
    
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
        col_min = df_copy[column].min()
        col_max = df_copy[column].max()
        if col_max != col_min:
            df_copy[f'{column}_normalized'] = (df_copy[column] - col_min) / (col_max - col_min)
        else:
            df_copy[f'{column}_normalized'] = 0.5
    
    elif method == 'zscore':
        col_mean = df_copy[column].mean()
        col_std = df_copy[column].std()
        if col_std > 0:
            df_copy[f'{column}_normalized'] = (df_copy[column] - col_mean) / col_std
        else:
            df_copy[f'{column}_normalized'] = 0
    
    else:
        raise ValueError("Method must be 'minmax' or 'zscore'")
    
    return df_copy

def handle_missing_values(df, column, strategy='mean'):
    """
    Handle missing values in a DataFrame column.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to process
    strategy (str): Strategy for handling missing values ('mean', 'median', 'mode', 'drop')
    
    Returns:
    pd.DataFrame: DataFrame with handled missing values
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    df_copy = df.copy()
    
    if strategy == 'mean':
        fill_value = df_copy[column].mean()
    elif strategy == 'median':
        fill_value = df_copy[column].median()
    elif strategy == 'mode':
        fill_value = df_copy[column].mode()[0] if not df_copy[column].mode().empty else np.nan
    elif strategy == 'drop':
        df_copy = df_copy.dropna(subset=[column])
        return df_copy
    else:
        raise ValueError("Strategy must be 'mean', 'median', 'mode', or 'drop'")
    
    df_copy[column] = df_copy[column].fillna(fill_value)
    return df_copyimport numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column):
    """
    Remove outliers using the Interquartile Range method.
    """
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data

def remove_outliers_zscore(data, column, threshold=3):
    """
    Remove outliers using Z-score method.
    """
    z_scores = np.abs(stats.zscore(data[column]))
    filtered_data = data[z_scores < threshold]
    return filtered_data

def normalize_minmax(data, column):
    """
    Normalize data using Min-Max scaling.
    """
    min_val = data[column].min()
    max_val = data[column].max()
    data[column + '_normalized'] = (data[column] - min_val) / (max_val - min_val)
    return data

def normalize_zscore(data, column):
    """
    Normalize data using Z-score standardization.
    """
    mean_val = data[column].mean()
    std_val = data[column].std()
    data[column + '_standardized'] = (data[column] - mean_val) / std_val
    return data

def clean_dataset(df, numeric_columns, outlier_method='iqr', normalize_method='minmax'):
    """
    Main function to clean dataset by removing outliers and normalizing numeric columns.
    """
    cleaned_df = df.copy()
    
    for col in numeric_columns:
        if outlier_method == 'iqr':
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
        elif outlier_method == 'zscore':
            cleaned_df = remove_outliers_zscore(cleaned_df, col)
    
    for col in numeric_columns:
        if normalize_method == 'minmax':
            cleaned_df = normalize_minmax(cleaned_df, col)
        elif normalize_method == 'zscore':
            cleaned_df = normalize_zscore(cleaned_df, col)
    
    return cleaned_df

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'feature1': np.random.normal(100, 15, 1000),
        'feature2': np.random.exponential(50, 1000),
        'category': np.random.choice(['A', 'B', 'C'], 1000)
    })
    
    numeric_cols = ['feature1', 'feature2']
    cleaned = clean_dataset(sample_data, numeric_cols, outlier_method='zscore', normalize_method='zscore')
    print(f"Original shape: {sample_data.shape}")
    print(f"Cleaned shape: {cleaned.shape}")
    print(cleaned.head())import pandas as pd
import numpy as np
from pathlib import Path

def load_data(filepath):
    """Load data from CSV file."""
    try:
        df = pd.read_csv(filepath)
        print(f"Loaded data with shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None

def remove_duplicates(df, subset=None):
    """Remove duplicate rows from DataFrame."""
    initial_count = len(df)
    df_clean = df.drop_duplicates(subset=subset, keep='first')
    removed = initial_count - len(df_clean)
    print(f"Removed {removed} duplicate rows.")
    return df_clean

def standardize_column(df, column_name, case='lower'):
    """Standardize text in a column to lower or upper case."""
    if column_name not in df.columns:
        print(f"Warning: Column '{column_name}' not found.")
        return df
    
    if case == 'lower':
        df[column_name] = df[column_name].astype(str).str.lower()
    elif case == 'upper':
        df[column_name] = df[column_name].astype(str).str.upper()
    
    print(f"Standardized column '{column_name}' to {case}case.")
    return df

def fill_missing_values(df, column_name, fill_value=np.nan):
    """Fill missing values in a column with specified value."""
    if column_name not in df.columns:
        print(f"Warning: Column '{column_name}' not found.")
        return df
    
    missing_count = df[column_name].isnull().sum()
    df[column_name] = df[column_name].fillna(fill_value)
    print(f"Filled {missing_count} missing values in column '{column_name}'.")
    return df

def clean_dataframe(df, config):
    """Apply multiple cleaning operations based on configuration."""
    df_clean = df.copy()
    
    if config.get('remove_duplicates'):
        subset = config.get('duplicate_subset')
        df_clean = remove_duplicates(df_clean, subset)
    
    for col_config in config.get('standardize_columns', []):
        df_clean = standardize_column(df_clean, **col_config)
    
    for col_config in config.get('fill_missing', []):
        df_clean = fill_missing_values(df_clean, **col_config)
    
    return df_clean

def save_cleaned_data(df, input_path, suffix='_cleaned'):
    """Save cleaned DataFrame to CSV file."""
    input_path = Path(input_path)
    output_path = input_path.parent / f"{input_path.stem}{suffix}{input_path.suffix}"
    
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to: {output_path}")
    return output_path

def main():
    """Main execution function."""
    input_file = 'raw_data.csv'
    output_suffix = '_processed'
    
    cleaning_config = {
        'remove_duplicates': True,
        'duplicate_subset': ['id', 'name'],
        'standardize_columns': [
            {'column_name': 'category', 'case': 'lower'},
            {'column_name': 'status', 'case': 'upper'}
        ],
        'fill_missing': [
            {'column_name': 'price', 'fill_value': 0},
            {'column_name': 'quantity', 'fill_value': 1}
        ]
    }
    
    raw_data = load_data(input_file)
    if raw_data is not None:
        cleaned_data = clean_dataframe(raw_data, cleaning_config)
        save_cleaned_data(cleaned_data, input_file, output_suffix)
        print("Data cleaning completed successfully.")

if __name__ == "__main__":
    main()import pandas as pd
import numpy as np
from typing import List, Optional

def remove_duplicates(df: pd.DataFrame, subset: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Remove duplicate rows from DataFrame.
    
    Args:
        df: Input DataFrame
        subset: Columns to consider for identifying duplicates
    
    Returns:
        DataFrame with duplicates removed
    """
    return df.drop_duplicates(subset=subset, keep='first')

def normalize_column(df: pd.DataFrame, column: str, method: str = 'minmax') -> pd.DataFrame:
    """
    Normalize specified column using given method.
    
    Args:
        df: Input DataFrame
        column: Column name to normalize
        method: Normalization method ('minmax' or 'zscore')
    
    Returns:
        DataFrame with normalized column
    """
    df_copy = df.copy()
    
    if method == 'minmax':
        min_val = df_copy[column].min()
        max_val = df_copy[column].max()
        if max_val > min_val:
            df_copy[column] = (df_copy[column] - min_val) / (max_val - min_val)
    
    elif method == 'zscore':
        mean_val = df_copy[column].mean()
        std_val = df_copy[column].std()
        if std_val > 0:
            df_copy[column] = (df_copy[column] - mean_val) / std_val
    
    return df_copy

def handle_missing_values(df: pd.DataFrame, strategy: str = 'mean') -> pd.DataFrame:
    """
    Handle missing values in numeric columns.
    
    Args:
        df: Input DataFrame
        strategy: Imputation strategy ('mean', 'median', or 'drop')
    
    Returns:
        DataFrame with handled missing values
    """
    df_copy = df.copy()
    numeric_cols = df_copy.select_dtypes(include=[np.number]).columns
    
    if strategy == 'drop':
        return df_copy.dropna(subset=numeric_cols)
    
    for col in numeric_cols:
        if strategy == 'mean':
            fill_value = df_copy[col].mean()
        elif strategy == 'median':
            fill_value = df_copy[col].median()
        else:
            fill_value = 0
        
        df_copy[col] = df_copy[col].fillna(fill_value)
    
    return df_copy

def clean_dataframe(df: pd.DataFrame, 
                   deduplicate: bool = True,
                   normalize_cols: Optional[List[str]] = None,
                   missing_strategy: str = 'mean') -> pd.DataFrame:
    """
    Comprehensive data cleaning pipeline.
    
    Args:
        df: Input DataFrame
        deduplicate: Whether to remove duplicates
        normalize_cols: Columns to normalize
        missing_strategy: Strategy for handling missing values
    
    Returns:
        Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    if deduplicate:
        cleaned_df = remove_duplicates(cleaned_df)
    
    cleaned_df = handle_missing_values(cleaned_df, strategy=missing_strategy)
    
    if normalize_cols:
        for col in normalize_cols:
            if col in cleaned_df.columns:
                cleaned_df = normalize_column(cleaned_df, col)
    
    return cleaned_df

def validate_dataframe(df: pd.DataFrame) -> bool:
    """
    Basic validation of DataFrame structure and content.
    
    Args:
        df: DataFrame to validate
    
    Returns:
        Boolean indicating if DataFrame is valid
    """
    if df.empty:
        return False
    
    if df.isnull().all().any():
        return False
    
    return True

if __name__ == "__main__":
    sample_data = {
        'id': [1, 2, 3, 4, 5, 5],
        'value': [10, 20, None, 40, 50, 50],
        'score': [100, 200, 300, 400, 500, 500]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    
    cleaned = clean_dataframe(df, normalize_cols=['value', 'score'])
    print("\nCleaned DataFrame:")
    print(cleaned)
    
    is_valid = validate_dataframe(cleaned)
    print(f"\nDataFrame is valid: {is_valid}")
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

def clean_dataset(file_path, columns_to_clean):
    df = pd.read_csv(file_path)
    
    for column in columns_to_clean:
        if column in df.columns:
            df = remove_outliers_iqr(df, column)
            df = normalize_minmax(df, column)
    
    cleaned_file_path = file_path.replace('.csv', '_cleaned.csv')
    df.to_csv(cleaned_file_path, index=False)
    return cleaned_file_path

if __name__ == "__main__":
    data_file = "sample_data.csv"
    columns = ["feature1", "feature2", "feature3"]
    result = clean_dataset(data_file, columns)
    print(f"Cleaned data saved to: {result}")
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
            
        if method == 'minmax':
            col_min = self.df[column].min()
            col_max = self.df[column].max()
            if col_max != col_min:
                self.df[f'{column}_normalized'] = (self.df[column] - col_min) / (col_max - col_min)
            else:
                self.df[f'{column}_normalized'] = 0
                
        elif method == 'zscore':
            mean_val = self.df[column].mean()
            std_val = self.df[column].std()
            if std_val > 0:
                self.df[f'{column}_normalized'] = (self.df[column] - mean_val) / std_val
            else:
                self.df[f'{column}_normalized'] = 0
                
        return self.df
    
    def fill_missing_values(self, strategy='mean', custom_value=None):
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if self.df[col].isnull().any():
                if strategy == 'mean':
                    fill_val = self.df[col].mean()
                elif strategy == 'median':
                    fill_val = self.df[col].median()
                elif strategy == 'mode':
                    fill_val = self.df[col].mode()[0]
                elif strategy == 'custom' and custom_value is not None:
                    fill_val = custom_value
                else:
                    continue
                    
                self.df[col] = self.df[col].fillna(fill_val)
                
        return self.df
    
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
        'feature_a': np.random.normal(100, 15, 100),
        'feature_b': np.random.exponential(50, 100),
        'feature_c': np.random.uniform(0, 1, 100)
    }
    
    df = pd.DataFrame(data)
    
    df.loc[10:15, 'feature_a'] = np.nan
    df.loc[95, 'feature_a'] = 500
    df.loc[96, 'feature_b'] = 1000
    
    return df

if __name__ == "__main__":
    sample_df = create_sample_data()
    cleaner = DataCleaner(sample_df)
    
    print("Original data shape:", cleaner.original_shape)
    print("Missing values before cleaning:", sample_df.isnull().sum().sum())
    
    removed = cleaner.remove_outliers_iqr(['feature_a', 'feature_b'])
    print(f"Removed {removed} outliers")
    
    cleaner.fill_missing_values(strategy='mean')
    print("Missing values after cleaning:", cleaner.df.isnull().sum().sum())
    
    cleaner.normalize_column('feature_a', method='zscore')
    cleaner.normalize_column('feature_c', method='minmax')
    
    summary = cleaner.get_summary()
    print("\nCleaning Summary:")
    for key, value in summary.items():
        print(f"{key}: {value}")
    
    cleaned_data = cleaner.get_cleaned_data()
    print(f"\nCleaned data shape: {cleaned_data.shape}")
    print("\nFirst 5 rows of cleaned data:")
    print(cleaned_data.head())
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def clean_dataset(file_path, output_path):
    df = pd.read_csv(file_path)
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_columns:
        df = remove_outliers_iqr(df, col)
    
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to {output_path}")
    return df

if __name__ == "__main__":
    input_file = "raw_data.csv"
    output_file = "cleaned_data.csv"
    cleaned_df = clean_dataset(input_file, output_file)