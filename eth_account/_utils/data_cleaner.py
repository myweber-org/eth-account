
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

def clean_dataset(file_path, output_path):
    df = pd.read_csv(file_path)
    
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_columns:
        df = remove_outliers_iqr(df, col)
    
    for col in numeric_columns:
        df = normalize_minmax(df, col)
    
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to {output_path}")
    return df

if __name__ == "__main__":
    input_file = "raw_data.csv"
    output_file = "cleaned_data.csv"
    cleaned_df = clean_dataset(input_file, output_file)import pandas as pd
import numpy as np
from typing import Optional, Dict, List

class DataCleaner:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.original_shape = df.shape
        
    def remove_duplicates(self, subset: Optional[List[str]] = None) -> 'DataCleaner':
        self.df = self.df.drop_duplicates(subset=subset)
        return self
        
    def handle_missing_values(self, 
                             strategy: str = 'mean',
                             custom_values: Optional[Dict] = None) -> 'DataCleaner':
        if strategy == 'drop':
            self.df = self.df.dropna()
        elif strategy == 'mean':
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            self.df[numeric_cols] = self.df[numeric_cols].fillna(self.df[numeric_cols].mean())
        elif strategy == 'median':
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            self.df[numeric_cols] = self.df[numeric_cols].fillna(self.df[numeric_cols].median())
        elif strategy == 'custom' and custom_values:
            self.df = self.df.fillna(custom_values)
        return self
        
    def convert_dtypes(self, 
                      date_columns: Optional[List[str]] = None,
                      categorical_columns: Optional[List[str]] = None) -> 'DataCleaner':
        if date_columns:
            for col in date_columns:
                if col in self.df.columns:
                    self.df[col] = pd.to_datetime(self.df[col], errors='coerce')
                    
        if categorical_columns:
            for col in categorical_columns:
                if col in self.df.columns:
                    self.df[col] = self.df[col].astype('category')
                    
        return self
        
    def remove_outliers(self, 
                       columns: List[str],
                       method: str = 'iqr',
                       threshold: float = 1.5) -> 'DataCleaner':
        for col in columns:
            if col not in self.df.columns or self.df[col].dtype not in [np.number]:
                continue
                
            if method == 'iqr':
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                self.df = self.df[(self.df[col] >= lower_bound) & (self.df[col] <= upper_bound)]
                
        return self
        
    def standardize_columns(self, 
                          columns: List[str],
                          method: str = 'zscore') -> 'DataCleaner':
        for col in columns:
            if col not in self.df.columns or self.df[col].dtype not in [np.number]:
                continue
                
            if method == 'zscore':
                self.df[col] = (self.df[col] - self.df[col].mean()) / self.df[col].std()
            elif method == 'minmax':
                self.df[col] = (self.df[col] - self.df[col].min()) / (self.df[col].max() - self.df[col].min())
                
        return self
        
    def get_cleaned_data(self) -> pd.DataFrame:
        return self.df
        
    def get_cleaning_report(self) -> Dict:
        return {
            'original_shape': self.original_shape,
            'cleaned_shape': self.df.shape,
            'rows_removed': self.original_shape[0] - self.df.shape[0],
            'columns_removed': self.original_shape[1] - self.df.shape[1],
            'missing_values_remaining': self.df.isnull().sum().sum(),
            'duplicates_remaining': self.df.duplicated().sum()
        }

def clean_csv_file(input_path: str,
                  output_path: str,
                  missing_strategy: str = 'mean',
                  date_columns: Optional[List[str]] = None,
                  outlier_columns: Optional[List[str]] = None) -> Dict:
    df = pd.read_csv(input_path)
    cleaner = DataCleaner(df)
    
    cleaner.remove_duplicates()
    cleaner.handle_missing_values(strategy=missing_strategy)
    
    if date_columns:
        cleaner.convert_dtypes(date_columns=date_columns)
        
    if outlier_columns:
        cleaner.remove_outliers(columns=outlier_columns)
        
    cleaned_df = cleaner.get_cleaned_data()
    cleaned_df.to_csv(output_path, index=False)
    
    return cleaner.get_cleaning_report()import pandas as pd
import numpy as np

def clean_dataset(df, duplicate_threshold=0.8, missing_strategy='median'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean
    duplicate_threshold (float): Threshold for considering rows as duplicates (0.0 to 1.0)
    missing_strategy (str): Strategy for handling missing values ('median', 'mean', 'drop', 'fill')
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    
    original_shape = df.shape
    print(f"Original dataset shape: {original_shape}")
    
    # Remove exact duplicates
    df_cleaned = df.drop_duplicates()
    exact_duplicates = original_shape[0] - df_cleaned.shape[0]
    print(f"Removed {exact_duplicates} exact duplicate rows")
    
    # Remove approximate duplicates based on threshold
    if duplicate_threshold < 1.0:
        numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            similarity_matrix = df_cleaned[numeric_cols].corr().abs()
            high_correlation = (similarity_matrix > duplicate_threshold) & (similarity_matrix < 1.0)
            duplicate_pairs = np.where(high_correlation)
            
            if len(duplicate_pairs[0]) > 0:
                cols_to_drop = set()
                for i, j in zip(duplicate_pairs[0], duplicate_pairs[1]):
                    if i < j:
                        cols_to_drop.add(df_cleaned.columns[j])
                
                df_cleaned = df_cleaned.drop(columns=list(cols_to_drop))
                print(f"Removed {len(cols_to_drop)} highly correlated columns")
    
    # Handle missing values
    missing_before = df_cleaned.isnull().sum().sum()
    
    if missing_strategy == 'drop':
        df_cleaned = df_cleaned.dropna()
        print(f"Dropped rows with missing values")
    elif missing_strategy == 'median':
        numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df_cleaned[col].isnull().any():
                df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].median())
    elif missing_strategy == 'mean':
        numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df_cleaned[col].isnull().any():
                df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].mean())
    elif missing_strategy == 'fill':
        df_cleaned = df_cleaned.fillna(method='ffill').fillna(method='bfill')
    
    missing_after = df_cleaned.isnull().sum().sum()
    print(f"Handled {missing_before - missing_after} missing values")
    
    # Remove constant columns
    constant_cols = [col for col in df_cleaned.columns if df_cleaned[col].nunique() <= 1]
    if constant_cols:
        df_cleaned = df_cleaned.drop(columns=constant_cols)
        print(f"Removed {len(constant_cols)} constant columns: {constant_cols}")
    
    final_shape = df_cleaned.shape
    print(f"Final dataset shape: {final_shape}")
    print(f"Reduced from {original_shape[0]} to {final_shape[0]} rows "
          f"and {original_shape[1]} to {final_shape[1]} columns")
    
    return df_cleaned

def validate_dataframe(df, min_rows=10, required_columns=None):
    """
    Validate that a DataFrame meets basic requirements.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate
    min_rows (int): Minimum number of rows required
    required_columns (list): List of column names that must be present
    
    Returns:
    bool: True if validation passes, False otherwise
    """
    
    if df.empty:
        print("Validation failed: DataFrame is empty")
        return False
    
    if len(df) < min_rows:
        print(f"Validation failed: Less than {min_rows} rows")
        return False
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            print(f"Validation failed: Missing required columns: {missing_cols}")
            return False
    
    # Check for infinite values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        inf_count = np.isinf(df[numeric_cols]).sum().sum()
        if inf_count > 0:
            print(f"Warning: Found {inf_count} infinite values")
    
    print("DataFrame validation passed")
    return True

# Example usage
if __name__ == "__main__":
    # Create sample data
    sample_data = {
        'A': [1, 2, 2, 4, 5, None, 7, 8, 9, 10],
        'B': [1.1, 2.2, 2.2, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.0],
        'C': ['a', 'b', 'b', 'd', 'e', 'f', 'g', 'h', 'i', 'j'],
        'D': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  # Constant column
    }
    
    df = pd.DataFrame(sample_data)
    
    # Clean the dataset
    cleaned_df = clean_dataset(df, duplicate_threshold=0.9, missing_strategy='median')
    
    # Validate the cleaned dataset
    is_valid = validate_dataframe(cleaned_df, min_rows=5, required_columns=['A', 'B', 'C'])
    
    print(f"\nCleaned DataFrame head:")
    print(cleaned_df.head())
import pandas as pd
import numpy as np

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
    
    return filtered_df

def calculate_summary_statistics(df, column):
    """
    Calculate summary statistics for a DataFrame column.
    
    Args:
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

def clean_dataset(df, numeric_columns=None):
    """
    Clean a dataset by removing outliers from all numeric columns.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        numeric_columns (list, optional): List of numeric columns to clean.
                                         If None, uses all numeric columns.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_df = df.copy()
    
    for column in numeric_columns:
        if column in df.columns:
            original_count = len(cleaned_df)
            cleaned_df = remove_outliers_iqr(cleaned_df, column)
            removed_count = original_count - len(cleaned_df)
            print(f"Removed {removed_count} outliers from column '{column}'")
    
    return cleaned_dfimport numpy as np
import pandas as pd
from scipy import stats

def detect_outliers_iqr(data, column, threshold=1.5):
    """
    Detect outliers using IQR method
    """
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr
    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    return outliers, lower_bound, upper_bound

def remove_outliers(data, column, threshold=1.5):
    """
    Remove outliers from specified column
    """
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data

def normalize_minmax(data, column):
    """
    Normalize data using min-max scaling
    """
    min_val = data[column].min()
    max_val = data[column].max()
    normalized = (data[column] - min_val) / (max_val - min_val)
    return normalized

def standardize_zscore(data, column):
    """
    Standardize data using z-score normalization
    """
    mean_val = data[column].mean()
    std_val = data[column].std()
    standardized = (data[column] - mean_val) / std_val
    return standardized

def handle_missing_values(data, strategy='mean', columns=None):
    """
    Handle missing values with specified strategy
    """
    data_copy = data.copy()
    
    if columns is None:
        columns = data_copy.select_dtypes(include=[np.number]).columns
    
    for column in columns:
        if strategy == 'mean':
            fill_value = data_copy[column].mean()
        elif strategy == 'median':
            fill_value = data_copy[column].median()
        elif strategy == 'mode':
            fill_value = data_copy[column].mode()[0]
        elif strategy == 'drop':
            data_copy = data_copy.dropna(subset=[column])
            continue
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        data_copy[column] = data_copy[column].fillna(fill_value)
    
    return data_copy

def clean_dataset(data, outlier_columns=None, normalize_columns=None, 
                  standardize_columns=None, missing_strategy='mean'):
    """
    Comprehensive data cleaning pipeline
    """
    cleaned_data = data.copy()
    
    # Handle missing values
    cleaned_data = handle_missing_values(cleaned_data, strategy=missing_strategy)
    
    # Remove outliers
    if outlier_columns:
        for column in outlier_columns:
            cleaned_data = remove_outliers(cleaned_data, column)
    
    # Normalize specified columns
    if normalize_columns:
        for column in normalize_columns:
            cleaned_data[f'{column}_normalized'] = normalize_minmax(cleaned_data, column)
    
    # Standardize specified columns
    if standardize_columns:
        for column in standardize_columns:
            cleaned_data[f'{column}_standardized'] = standardize_zscore(cleaned_data, column)
    
    return cleaned_data

def get_data_summary(data):
    """
    Generate comprehensive data summary
    """
    summary = {
        'shape': data.shape,
        'dtypes': data.dtypes.to_dict(),
        'missing_values': data.isnull().sum().to_dict(),
        'numeric_summary': data.describe().to_dict() if data.select_dtypes(include=[np.number]).shape[1] > 0 else {},
        'categorical_summary': data.select_dtypes(include=['object']).describe().to_dict() if data.select_dtypes(include=['object']).shape[1] > 0 else {}
    }
    return summary
import pandas as pd
import numpy as np

def clean_dataframe(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Parameters:
    df (pd.DataFrame): The DataFrame to clean.
    drop_duplicates (bool): If True, drop duplicate rows.
    fill_missing (str): Method to fill missing values. Options: 'mean', 'median', 'mode', or 'drop'.
    
    Returns:
    pd.DataFrame: The cleaned DataFrame.
    """
    df_clean = df.copy()
    
    if drop_duplicates:
        df_clean = df_clean.drop_duplicates()
    
    if fill_missing == 'drop':
        df_clean = df_clean.dropna()
    elif fill_missing in ['mean', 'median']:
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if fill_missing == 'mean':
                fill_value = df_clean[col].mean()
            else:
                fill_value = df_clean[col].median()
            df_clean[col].fillna(fill_value, inplace=True)
    elif fill_missing == 'mode':
        for col in df_clean.columns:
            mode_value = df_clean[col].mode()
            if not mode_value.empty:
                df_clean[col].fillna(mode_value[0], inplace=True)
    
    return df_clean

def validate_dataframe(df, check_nulls=True, check_types=True):
    """
    Validate a DataFrame for common data quality issues.
    
    Parameters:
    df (pd.DataFrame): The DataFrame to validate.
    check_nulls (bool): Check for null values.
    check_types (bool): Check for consistent data types.
    
    Returns:
    dict: A dictionary with validation results.
    """
    validation_results = {}
    
    if check_nulls:
        null_counts = df.isnull().sum()
        validation_results['null_counts'] = null_counts[null_counts > 0].to_dict()
    
    if check_types:
        dtypes = df.dtypes.to_dict()
        validation_results['dtypes'] = dtypes
    
    validation_results['shape'] = df.shape
    validation_results['columns'] = list(df.columns)
    
    return validation_results

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, 2, 4, None],
        'B': [5, None, 7, 8, 9],
        'C': ['x', 'y', 'y', 'z', 'z']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    
    cleaned_df = clean_dataframe(df, fill_missing='mean')
    print("\nCleaned DataFrame:")
    print(cleaned_df)
    
    validation = validate_dataframe(cleaned_df)
    print("\nValidation Results:")
    for key, value in validation.items():
        print(f"{key}: {value}")
import pandas as pd
import re

def clean_dataframe(df, column_mapping=None, drop_duplicates=True, normalize_text=True):
    """
    Clean a pandas DataFrame by removing duplicates and normalizing text columns.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean
        column_mapping (dict, optional): Dictionary mapping old column names to new ones
        drop_duplicates (bool): Whether to remove duplicate rows
        normalize_text (bool): Whether to normalize text columns (strip, lower case)
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    if column_mapping:
        cleaned_df = cleaned_df.rename(columns=column_mapping)
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates().reset_index(drop=True)
    
    if normalize_text:
        for col in cleaned_df.select_dtypes(include=['object']).columns:
            cleaned_df[col] = cleaned_df[col].astype(str).str.strip().str.lower()
    
    return cleaned_df

def remove_special_characters(text, keep_pattern=r'[a-zA-Z0-9\s]'):
    """
    Remove special characters from text, keeping only alphanumeric and spaces by default.
    
    Args:
        text (str): Input text
        keep_pattern (str): Regex pattern of characters to keep
    
    Returns:
        str: Cleaned text
    """
    if pd.isna(text):
        return text
    
    text = str(text)
    return re.sub(f'[^{keep_pattern}]', '', text)

def validate_email(email):
    """
    Validate email format using regex pattern.
    
    Args:
        email (str): Email address to validate
    
    Returns:
        bool: True if email format is valid
    """
    if pd.isna(email):
        return False
    
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, str(email).strip()))

def standardize_date(date_str, target_format='%Y-%m-%d'):
    """
    Attempt to standardize date strings to a common format.
    
    Args:
        date_str (str): Date string in various formats
        target_format (str): Target date format
    
    Returns:
        str: Standardized date string or original if parsing fails
    """
    if pd.isna(date_str):
        return date_str
    
    date_str = str(date_str).strip()
    
    common_formats = [
        '%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y',
        '%d-%m-%Y', '%m-%d-%Y', '%Y/%m/%d',
        '%b %d, %Y', '%B %d, %Y', '%d %b %Y'
    ]
    
    for fmt in common_formats:
        try:
            parsed_date = pd.to_datetime(date_str, format=fmt)
            return parsed_date.strftime(target_format)
        except (ValueError, TypeError):
            continue
    
    try:
        parsed_date = pd.to_datetime(date_str)
        return parsed_date.strftime(target_format)
    except (ValueError, TypeError):
        return date_str