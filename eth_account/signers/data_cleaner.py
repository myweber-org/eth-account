import pandas as pd
import numpy as np

def clean_dataset(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by handling duplicates and missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    drop_duplicates (bool): Whether to drop duplicate rows.
    fill_missing (str): Method to fill missing values ('mean', 'median', 'mode', or 'drop').
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    if fill_missing == 'drop':
        cleaned_df = cleaned_df.dropna()
    else:
        for column in cleaned_df.select_dtypes(include=[np.number]).columns:
            if cleaned_df[column].isnull().any():
                if fill_missing == 'mean':
                    cleaned_df[column].fillna(cleaned_df[column].mean(), inplace=True)
                elif fill_missing == 'median':
                    cleaned_df[column].fillna(cleaned_df[column].median(), inplace=True)
                elif fill_missing == 'mode':
                    cleaned_df[column].fillna(cleaned_df[column].mode()[0], inplace=True)
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of column names that must be present.
    
    Returns:
    tuple: (is_valid, error_message)
    """
    if not isinstance(df, pd.DataFrame):
        return False, "Input is not a pandas DataFrame"
    
    if df.empty:
        return False, "DataFrame is empty"
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
    
    return True, "DataFrame is valid"

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, 2, 3, None],
        'B': [4, None, 6, 6, 8],
        'C': [7, 8, 9, 10, 11]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    
    cleaned = clean_dataset(df, drop_duplicates=True, fill_missing='mean')
    print("\nCleaned DataFrame:")
    print(cleaned)
    
    is_valid, message = validate_dataframe(cleaned, required_columns=['A', 'B', 'C'])
    print(f"\nValidation: {is_valid} - {message}")
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Union

class DataCleaner:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.original_shape = df.shape
        
    def remove_duplicates(self, subset: Optional[List[str]] = None) -> 'DataCleaner':
        self.df = self.df.drop_duplicates(subset=subset, keep='first')
        return self
        
    def fill_missing_values(self, strategy: str = 'mean', 
                           columns: Optional[List[str]] = None,
                           custom_value: Optional[Union[int, float, str]] = None) -> 'DataCleaner':
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns.tolist()
            
        for col in columns:
            if col in self.df.columns:
                if strategy == 'mean':
                    self.df[col] = self.df[col].fillna(self.df[col].mean())
                elif strategy == 'median':
                    self.df[col] = self.df[col].fillna(self.df[col].median())
                elif strategy == 'mode':
                    self.df[col] = self.df[col].fillna(self.df[col].mode()[0])
                elif strategy == 'custom' and custom_value is not None:
                    self.df[col] = self.df[col].fillna(custom_value)
                elif strategy == 'ffill':
                    self.df[col] = self.df[col].fillna(method='ffill')
                elif strategy == 'bfill':
                    self.df[col] = self.df[col].fillna(method='bfill')
                    
        return self
        
    def remove_outliers(self, columns: List[str], 
                       method: str = 'iqr',
                       threshold: float = 1.5) -> 'DataCleaner':
        for col in columns:
            if col in self.df.columns and self.df[col].dtype in [np.float64, np.int64]:
                if method == 'iqr':
                    Q1 = self.df[col].quantile(0.25)
                    Q3 = self.df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - threshold * IQR
                    upper_bound = Q3 + threshold * IQR
                    self.df = self.df[(self.df[col] >= lower_bound) & (self.df[col] <= upper_bound)]
                elif method == 'zscore':
                    from scipy import stats
                    z_scores = np.abs(stats.zscore(self.df[col].dropna()))
                    mask = z_scores < threshold
                    valid_indices = self.df[col].dropna().index[mask]
                    self.df = self.df.loc[valid_indices]
                    
        return self
        
    def normalize_columns(self, columns: List[str], 
                         method: str = 'minmax') -> 'DataCleaner':
        for col in columns:
            if col in self.df.columns and self.df[col].dtype in [np.float64, np.int64]:
                if method == 'minmax':
                    min_val = self.df[col].min()
                    max_val = self.df[col].max()
                    if max_val != min_val:
                        self.df[col] = (self.df[col] - min_val) / (max_val - min_val)
                elif method == 'zscore':
                    mean_val = self.df[col].mean()
                    std_val = self.df[col].std()
                    if std_val != 0:
                        self.df[col] = (self.df[col] - mean_val) / std_val
                        
        return self
        
    def get_cleaning_report(self) -> Dict:
        report = {
            'original_rows': self.original_shape[0],
            'original_columns': self.original_shape[1],
            'cleaned_rows': self.df.shape[0],
            'cleaned_columns': self.df.shape[1],
            'rows_removed': self.original_shape[0] - self.df.shape[0],
            'missing_values': self.df.isnull().sum().sum(),
            'duplicates_removed': self.original_shape[0] - self.df.shape[0]
        }
        return report
        
    def get_dataframe(self) -> pd.DataFrame:
        return self.df.copy()
        
    def save_cleaned_data(self, filepath: str, format: str = 'csv'):
        if format == 'csv':
            self.df.to_csv(filepath, index=False)
        elif format == 'excel':
            self.df.to_excel(filepath, index=False)
        elif format == 'parquet':
            self.df.to_parquet(filepath, index=False)

def load_and_clean_csv(filepath: str, 
                      fill_strategy: str = 'mean',
                      remove_outliers: bool = True) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    cleaner = DataCleaner(df)
    
    cleaner.remove_duplicates()
    cleaner.fill_missing_values(strategy=fill_strategy)
    
    if remove_outliers:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            cleaner.remove_outliers(numeric_cols)
    
    report = cleaner.get_cleaning_report()
    print(f"Cleaning completed. Removed {report['rows_removed']} rows.")
    
    return cleaner.get_dataframe()import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the IQR method.
    
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

def clean_numeric_data(df, columns=None):
    """
    Clean numeric data by removing outliers from specified columns.
    If no columns specified, clean all numeric columns.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    columns (list): List of column names to clean
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    if columns is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        columns = list(numeric_cols)
    
    cleaned_df = df.copy()
    
    for col in columns:
        if col in cleaned_df.columns:
            original_len = len(cleaned_df)
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
            removed_count = original_len - len(cleaned_df)
            print(f"Removed {removed_count} outliers from column '{col}'")
    
    return cleaned_df

def validate_dataframe(df):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate
    
    Returns:
    dict: Validation results
    """
    validation_results = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': df.isnull().sum().sum(),
        'duplicate_rows': df.duplicated().sum(),
        'data_types': df.dtypes.to_dict()
    }
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        validation_results['numeric_stats'] = df[numeric_cols].describe().to_dict()
    
    return validation_results

if __name__ == "__main__":
    sample_data = {
        'id': range(1, 101),
        'value': np.random.randn(100) * 10 + 50,
        'category': np.random.choice(['A', 'B', 'C'], 100)
    }
    
    df = pd.DataFrame(sample_data)
    df.loc[0, 'value'] = 200
    df.loc[1, 'value'] = -100
    
    print("Original DataFrame:")
    print(df.head())
    print(f"\nOriginal shape: {df.shape}")
    
    validation = validate_dataframe(df)
    print(f"\nValidation results: {validation}")
    
    cleaned_df = clean_numeric_data(df, columns=['value'])
    print(f"\nCleaned shape: {cleaned_df.shape}")
    
    print("\nCleaned DataFrame head:")
    print(cleaned_df.head())
import pandas as pd
import numpy as np

def clean_dataframe(df, missing_strategy='mean', outlier_threshold=3):
    """
    Clean a pandas DataFrame by handling missing values and outliers.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    missing_strategy (str): Strategy for handling missing values. 
                            Options: 'mean', 'median', 'drop', 'fill_zero'.
    outlier_threshold (float): Number of standard deviations for outlier detection.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    
    df_clean = df.copy()
    
    # Handle missing values
    if missing_strategy == 'mean':
        df_clean = df_clean.fillna(df_clean.mean())
    elif missing_strategy == 'median':
        df_clean = df_clean.fillna(df_clean.median())
    elif missing_strategy == 'drop':
        df_clean = df_clean.dropna()
    elif missing_strategy == 'fill_zero':
        df_clean = df_clean.fillna(0)
    else:
        raise ValueError(f"Unknown missing strategy: {missing_strategy}")
    
    # Remove outliers using z-score method
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        z_scores = np.abs((df_clean[col] - df_clean[col].mean()) / df_clean[col].std())
        df_clean = df_clean[z_scores < outlier_threshold]
    
    # Reset index after removing outliers
    df_clean = df_clean.reset_index(drop=True)
    
    return df_clean

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of column names that must be present.
    
    Returns:
    tuple: (is_valid, error_message)
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

# Example usage
if __name__ == "__main__":
    # Create sample data with missing values and outliers
    np.random.seed(42)
    sample_data = {
        'feature_a': np.random.randn(100),
        'feature_b': np.random.randn(100),
        'feature_c': np.random.randn(100)
    }
    
    # Introduce missing values
    sample_df = pd.DataFrame(sample_data)
    sample_df.loc[10:15, 'feature_a'] = np.nan
    sample_df.loc[20:25, 'feature_b'] = np.nan
    
    # Introduce outliers
    sample_df.loc[0, 'feature_c'] = 100
    
    print("Original DataFrame shape:", sample_df.shape)
    print("Missing values:\n", sample_df.isnull().sum())
    
    # Clean the data
    cleaned_df = clean_dataframe(sample_df, missing_strategy='mean', outlier_threshold=3)
    
    print("\nCleaned DataFrame shape:", cleaned_df.shape)
    print("Missing values after cleaning:\n", cleaned_df.isnull().sum())
    
    # Validate the cleaned data
    is_valid, message = validate_dataframe(cleaned_df, required_columns=['feature_a', 'feature_b', 'feature_c'])
    print(f"\nValidation: {is_valid} - {message}")
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

def clean_dataset(df, numeric_columns):
    cleaned_df = df.copy()
    
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
            cleaned_df = normalize_minmax(cleaned_df, col)
    
    cleaned_df = cleaned_df.dropna()
    cleaned_df = cleaned_df.reset_index(drop=True)
    
    return cleaned_df

def calculate_statistics(df, column):
    if column not in df.columns:
        return None
    
    stats_dict = {
        'mean': df[column].mean(),
        'median': df[column].median(),
        'std': df[column].std(),
        'variance': df[column].var(),
        'skewness': df[column].skew(),
        'kurtosis': df[column].kurtosis()
    }
    
    return stats_dict

if __name__ == "__main__":
    sample_data = {
        'A': np.random.normal(100, 15, 1000),
        'B': np.random.exponential(50, 1000),
        'C': np.random.uniform(0, 200, 1000)
    }
    
    df = pd.DataFrame(sample_data)
    print("Original dataset shape:", df.shape)
    
    numeric_cols = ['A', 'B', 'C']
    cleaned_df = clean_dataset(df, numeric_cols)
    print("Cleaned dataset shape:", cleaned_df.shape)
    
    for col in numeric_cols:
        stats_result = calculate_statistics(cleaned_df, col)
        print(f"\nStatistics for {col}:")
        for key, value in stats_result.items():
            print(f"{key}: {value:.4f}")