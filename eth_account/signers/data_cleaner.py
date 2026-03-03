import pandas as pd
import numpy as np

def clean_dataset(df, drop_duplicates=True, fill_missing='mean'):
    """
    Cleans a pandas DataFrame by removing duplicates and handling missing values.
    """
    original_shape = df.shape
    cleaned_df = df.copy()

    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
        print(f"Removed {original_shape[0] - cleaned_df.shape[0]} duplicate rows.")

    if fill_missing:
        numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if cleaned_df[col].isnull().any():
                if fill_missing == 'mean':
                    fill_value = cleaned_df[col].mean()
                elif fill_missing == 'median':
                    fill_value = cleaned_df[col].median()
                elif fill_missing == 'mode':
                    fill_value = cleaned_df[col].mode()[0]
                else:
                    fill_value = fill_missing
                cleaned_df[col].fillna(fill_value, inplace=True)
                print(f"Filled missing values in column '{col}' with {fill_value}.")

    print(f"Original shape: {original_shape}, Cleaned shape: {cleaned_df.shape}")
    return cleaned_df

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, 2, np.nan, 5],
        'B': [10, np.nan, 10, 40, 50],
        'C': ['x', 'y', 'x', 'y', 'z']
    }
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    cleaned = clean_dataset(df, fill_missing='median')
    print("\nCleaned DataFrame:")
    print(cleaned)
import pandas as pd
import re

def clean_dataframe(df, column_mapping=None, drop_duplicates=True, normalize_text=True):
    """
    Clean a pandas DataFrame by removing duplicates and normalizing text columns.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean
        column_mapping (dict, optional): Dictionary to rename columns
        drop_duplicates (bool): Whether to remove duplicate rows
        normalize_text (bool): Whether to normalize text columns
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    if column_mapping:
        cleaned_df = cleaned_df.rename(columns=column_mapping)
    
    if drop_duplicates:
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates().reset_index(drop=True)
        removed = initial_rows - len(cleaned_df)
        print(f"Removed {removed} duplicate rows")
    
    if normalize_text:
        text_columns = cleaned_df.select_dtypes(include=['object']).columns
        for col in text_columns:
            cleaned_df[col] = cleaned_df[col].apply(_normalize_string)
    
    return cleaned_df

def _normalize_string(text):
    """
    Normalize a string by converting to lowercase, removing extra whitespace,
    and stripping special characters.
    """
    if pd.isna(text):
        return text
    
    text = str(text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    text = re.sub(r'[^\w\s-]', '', text)
    
    return text

def validate_email_column(df, email_column):
    """
    Validate email addresses in a DataFrame column.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        email_column (str): Name of the column containing email addresses
    
    Returns:
        pd.DataFrame: DataFrame with validation results
    """
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    
    validation_results = df.copy()
    validation_results['is_valid_email'] = validation_results[email_column].apply(
        lambda x: bool(re.match(email_pattern, str(x))) if pd.notna(x) else False
    )
    
    valid_count = validation_results['is_valid_email'].sum()
    total_count = len(validation_results)
    
    print(f"Valid emails: {valid_count}/{total_count} ({valid_count/total_count*100:.1f}%)")
    
    return validation_results
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column, threshold=1.5):
    """
    Remove outliers using IQR method
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data

def remove_outliers_zscore(data, column, threshold=3):
    """
    Remove outliers using Z-score method
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    z_scores = np.abs(stats.zscore(data[column]))
    filtered_data = data[z_scores < threshold]
    return filtered_data

def normalize_minmax(data, columns=None):
    """
    Normalize data using Min-Max scaling
    """
    if columns is None:
        columns = data.select_dtypes(include=[np.number]).columns
    
    normalized_data = data.copy()
    for col in columns:
        if col in data.columns and np.issubdtype(data[col].dtype, np.number):
            min_val = data[col].min()
            max_val = data[col].max()
            if max_val != min_val:
                normalized_data[col] = (data[col] - min_val) / (max_val - min_val)
    
    return normalized_data

def normalize_zscore(data, columns=None):
    """
    Normalize data using Z-score standardization
    """
    if columns is None:
        columns = data.select_dtypes(include=[np.number]).columns
    
    standardized_data = data.copy()
    for col in columns:
        if col in data.columns and np.issubdtype(data[col].dtype, np.number):
            mean_val = data[col].mean()
            std_val = data[col].std()
            if std_val != 0:
                standardized_data[col] = (data[col] - mean_val) / std_val
    
    return standardized_data

def clean_dataset(data, outlier_method='iqr', normalize_method='minmax', 
                  numeric_columns=None, outlier_threshold=None):
    """
    Comprehensive data cleaning pipeline
    """
    cleaned_data = data.copy()
    
    if numeric_columns is None:
        numeric_columns = cleaned_data.select_dtypes(include=[np.number]).columns.tolist()
    
    if outlier_method:
        for col in numeric_columns:
            if col in cleaned_data.columns:
                if outlier_method == 'iqr':
                    threshold = outlier_threshold if outlier_threshold else 1.5
                    cleaned_data = remove_outliers_iqr(cleaned_data, col, threshold)
                elif outlier_method == 'zscore':
                    threshold = outlier_threshold if outlier_threshold else 3
                    cleaned_data = remove_outliers_zscore(cleaned_data, col, threshold)
    
    if normalize_method:
        if normalize_method == 'minmax':
            cleaned_data = normalize_minmax(cleaned_data, numeric_columns)
        elif normalize_method == 'zscore':
            cleaned_data = normalize_zscore(cleaned_data, numeric_columns)
    
    return cleaned_data

def get_data_summary(data):
    """
    Generate summary statistics for the dataset
    """
    summary = {
        'original_rows': len(data),
        'numeric_columns': data.select_dtypes(include=[np.number]).columns.tolist(),
        'categorical_columns': data.select_dtypes(include=['object']).columns.tolist(),
        'missing_values': data.isnull().sum().to_dict(),
        'basic_stats': data.describe().to_dict()
    }
    return summary
import pandas as pd
import numpy as np

def clean_dataset(df, drop_duplicates=True, handle_nulls='drop', null_threshold=0.5):
    """
    Clean a pandas DataFrame by handling duplicates and null values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    drop_duplicates (bool): Whether to drop duplicate rows
    handle_nulls (str): Strategy for null handling - 'drop', 'fill', or 'threshold'
    null_threshold (float): Threshold for column null percentage when handle_nulls='threshold'
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        removed = initial_rows - len(cleaned_df)
        print(f"Removed {removed} duplicate rows")
    
    if handle_nulls == 'drop':
        cleaned_df = cleaned_df.dropna()
        print("Dropped rows with any null values")
    elif handle_nulls == 'fill':
        numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
        cleaned_df[numeric_cols] = cleaned_df[numeric_cols].fillna(cleaned_df[numeric_cols].median())
        non_numeric_cols = cleaned_df.select_dtypes(exclude=[np.number]).columns
        cleaned_df[non_numeric_cols] = cleaned_df[non_numeric_cols].fillna('Unknown')
        print("Filled null values with median (numeric) and 'Unknown' (non-numeric)")
    elif handle_nulls == 'threshold':
        null_percentages = cleaned_df.isnull().sum() / len(cleaned_df)
        cols_to_drop = null_percentages[null_percentages > null_threshold].index
        cleaned_df = cleaned_df.drop(columns=cols_to_drop)
        print(f"Dropped columns with >{null_threshold*100}% null values: {list(cols_to_drop)}")
    
    print(f"Original shape: {df.shape}, Cleaned shape: {cleaned_df.shape}")
    return cleaned_df

def validate_data(df, required_columns=None, min_rows=1):
    """
    Validate basic data structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate
    required_columns (list): List of required column names
    min_rows (int): Minimum number of rows required
    
    Returns:
    bool: True if validation passes, False otherwise
    """
    if len(df) < min_rows:
        print(f"Validation failed: DataFrame has only {len(df)} rows, minimum required is {min_rows}")
        return False
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            print(f"Validation failed: Missing required columns: {missing_cols}")
            return False
    
    if df.empty:
        print("Validation failed: DataFrame is empty")
        return False
    
    print("Data validation passed")
    return True

if __name__ == "__main__":
    sample_data = {
        'id': [1, 2, 2, 3, 4, 5],
        'name': ['Alice', 'Bob', 'Bob', None, 'Eve', None],
        'age': [25, 30, 30, None, 35, 40],
        'score': [85.5, 92.0, 92.0, 78.5, None, 88.0]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    cleaned = clean_dataset(df, handle_nulls='fill')
    print("\nCleaned DataFrame:")
    print(cleaned)
    
    is_valid = validate_data(cleaned, required_columns=['id', 'name'], min_rows=3)
    print(f"\nData is valid: {is_valid}")
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column, threshold=1.5):
    """
    Remove outliers using Interquartile Range method.
    
    Args:
        data: pandas DataFrame
        column: column name to process
        threshold: IQR multiplier (default 1.5)
    
    Returns:
        DataFrame with outliers removed
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data.copy()

def remove_outliers_zscore(data, column, threshold=3):
    """
    Remove outliers using Z-score method.
    
    Args:
        data: pandas DataFrame
        column: column name to process
        threshold: Z-score threshold (default 3)
    
    Returns:
        DataFrame with outliers removed
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    z_scores = np.abs(stats.zscore(data[column].dropna()))
    filtered_indices = np.where(z_scores < threshold)[0]
    filtered_data = data.iloc[filtered_indices].copy()
    return filtered_data

def normalize_minmax(data, column):
    """
    Normalize data using Min-Max scaling.
    
    Args:
        data: pandas DataFrame
        column: column name to normalize
    
    Returns:
        Series with normalized values
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    min_val = data[column].min()
    max_val = data[column].max()
    
    if max_val == min_val:
        return pd.Series([0.5] * len(data), index=data.index)
    
    normalized = (data[column] - min_val) / (max_val - min_val)
    return normalized

def normalize_zscore(data, column):
    """
    Normalize data using Z-score standardization.
    
    Args:
        data: pandas DataFrame
        column: column name to normalize
    
    Returns:
        Series with standardized values
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    mean_val = data[column].mean()
    std_val = data[column].std()
    
    if std_val == 0:
        return pd.Series([0] * len(data), index=data.index)
    
    standardized = (data[column] - mean_val) / std_val
    return standardized

def clean_dataset(data, numeric_columns=None, outlier_method='iqr', normalize_method='minmax'):
    """
    Comprehensive data cleaning pipeline.
    
    Args:
        data: pandas DataFrame
        numeric_columns: list of numeric columns to process (default: all numeric)
        outlier_method: 'iqr', 'zscore', or None
        normalize_method: 'minmax', 'zscore', or None
    
    Returns:
        Cleaned DataFrame
    """
    cleaned_data = data.copy()
    
    if numeric_columns is None:
        numeric_columns = cleaned_data.select_dtypes(include=[np.number]).columns.tolist()
    
    for column in numeric_columns:
        if column not in cleaned_data.columns:
            continue
            
        if outlier_method == 'iqr':
            cleaned_data = remove_outliers_iqr(cleaned_data, column)
        elif outlier_method == 'zscore':
            cleaned_data = remove_outliers_zscore(cleaned_data, column)
        
        if normalize_method == 'minmax':
            cleaned_data[f'{column}_normalized'] = normalize_minmax(cleaned_data, column)
        elif normalize_method == 'zscore':
            cleaned_data[f'{column}_standardized'] = normalize_zscore(cleaned_data, column)
    
    return cleaned_data

def get_summary_statistics(data):
    """
    Generate summary statistics for numeric columns.
    
    Args:
        data: pandas DataFrame
    
    Returns:
        DataFrame with summary statistics
    """
    numeric_data = data.select_dtypes(include=[np.number])
    
    if numeric_data.empty:
        return pd.DataFrame()
    
    summary = numeric_data.describe().T
    summary['skewness'] = numeric_data.skew()
    summary['kurtosis'] = numeric_data.kurtosis()
    summary['missing'] = numeric_data.isnull().sum()
    summary['missing_pct'] = (summary['missing'] / len(data)) * 100
    
    return summary

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'feature1': np.random.normal(100, 15, 1000),
        'feature2': np.random.exponential(50, 1000),
        'category': np.random.choice(['A', 'B', 'C'], 1000)
    })
    
    print("Original data shape:", sample_data.shape)
    print("\nOriginal summary statistics:")
    print(get_summary_statistics(sample_data))
    
    cleaned = clean_dataset(sample_data, outlier_method='iqr', normalize_method='minmax')
    print("\nCleaned data shape:", cleaned.shape)
    print("\nCleaned summary statistics:")
    print(get_summary_statistics(cleaned))
import numpy as np
import pandas as pd

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the IQR method.
    
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
    Calculate summary statistics for a column.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name
    
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

def example_usage():
    """
    Example usage of the data cleaning functions.
    """
    np.random.seed(42)
    
    data = {
        'id': range(100),
        'value': np.random.normal(100, 15, 100)
    }
    
    df = pd.DataFrame(data)
    
    print("Original DataFrame shape:", df.shape)
    print("Original statistics:", calculate_summary_statistics(df, 'value'))
    
    cleaned_df = remove_outliers_iqr(df, 'value')
    
    print("\nCleaned DataFrame shape:", cleaned_df.shape)
    print("Cleaned statistics:", calculate_summary_statistics(cleaned_df, 'value'))
    
    return cleaned_df

if __name__ == "__main__":
    cleaned_data = example_usage()
def remove_duplicates(sequence):
    seen = set()
    result = []
    for item in sequence:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result
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
    
    def normalize_data(self, columns=None, method='zscore'):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
            
        df_normalized = self.df.copy()
        for col in columns:
            if col in self.df.columns and pd.api.types.is_numeric_dtype(self.df[col]):
                if method == 'zscore':
                    df_normalized[col] = stats.zscore(self.df[col])
                elif method == 'minmax':
                    min_val = self.df[col].min()
                    max_val = self.df[col].max()
                    if max_val != min_val:
                        df_normalized[col] = (self.df[col] - min_val) / (max_val - min_val)
                    else:
                        df_normalized[col] = 0
                elif method == 'robust':
                    median = self.df[col].median()
                    iqr = self.df[col].quantile(0.75) - self.df[col].quantile(0.25)
                    if iqr != 0:
                        df_normalized[col] = (self.df[col] - median) / iqr
                    else:
                        df_normalized[col] = 0
                        
        self.df = df_normalized
        return self.df
    
    def handle_missing_values(self, strategy='mean', columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
            
        df_filled = self.df.copy()
        for col in columns:
            if col in self.df.columns and self.df[col].isnull().any():
                if strategy == 'mean':
                    fill_value = self.df[col].mean()
                elif strategy == 'median':
                    fill_value = self.df[col].median()
                elif strategy == 'mode':
                    fill_value = self.df[col].mode()[0]
                elif strategy == 'drop':
                    df_filled = df_filled.dropna(subset=[col])
                    continue
                    
                df_filled[col] = self.df[col].fillna(fill_value)
                
        self.df = df_filled.reset_index(drop=True)
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
            'numeric_columns': list(self.df.select_dtypes(include=[np.number]).columns),
            'categorical_columns': list(self.df.select_dtypes(include=['object', 'category']).columns)
        }
        return summary

def create_sample_data():
    np.random.seed(42)
    data = {
        'feature_a': np.random.normal(100, 15, 100),
        'feature_b': np.random.exponential(50, 100),
        'feature_c': np.random.randint(1, 100, 100),
        'category': np.random.choice(['A', 'B', 'C'], 100)
    }
    
    data['feature_a'][[10, 25, 50]] = np.nan
    data['feature_b'][[15, 30, 75]] = 1000
    
    return pd.DataFrame(data)

if __name__ == "__main__":
    sample_df = create_sample_data()
    cleaner = DataCleaner(sample_df)
    
    print("Original data shape:", cleaner.original_shape)
    print("\nHandling missing values...")
    cleaner.handle_missing_values(strategy='mean')
    
    print("\nRemoving outliers...")
    outliers_removed = cleaner.remove_outliers_iqr()
    print(f"Removed {outliers_removed} outliers")
    
    print("\nNormalizing data...")
    cleaner.normalize_data(method='zscore')
    
    print("\nCleaning summary:")
    summary = cleaner.get_summary()
    for key, value in summary.items():
        print(f"{key}: {value}")
    
    cleaned_data = cleaner.get_cleaned_data()
    print(f"\nFirst 5 rows of cleaned data:\n{cleaned_data.head()}")