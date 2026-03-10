
import numpy as np
import pandas as pd

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
    
    return filtered_df.copy()

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
        'count': df[column].count(),
        'q1': df[column].quantile(0.25),
        'q3': df[column].quantile(0.75)
    }
    
    return stats

def clean_dataset(df, numeric_columns=None):
    """
    Clean a dataset by removing outliers from all numeric columns.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    numeric_columns (list): List of numeric column names to clean
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_df = df.copy()
    
    for column in numeric_columns:
        if column in df.columns and pd.api.types.is_numeric_dtype(df[column]):
            try:
                cleaned_df = remove_outliers_iqr(cleaned_df, column)
            except Exception as e:
                print(f"Warning: Could not process column '{column}': {e}")
                continue
    
    return cleaned_df

if __name__ == "__main__":
    sample_data = {
        'id': range(1, 101),
        'value': np.random.normal(100, 15, 100),
        'category': np.random.choice(['A', 'B', 'C'], 100)
    }
    
    sample_data['value'][0] = 500
    sample_data['value'][1] = -200
    
    df = pd.DataFrame(sample_data)
    
    print("Original dataset shape:", df.shape)
    print("Original statistics:", calculate_summary_statistics(df, 'value'))
    
    cleaned_df = clean_dataset(df, ['value'])
    
    print("\nCleaned dataset shape:", cleaned_df.shape)
    print("Cleaned statistics:", calculate_summary_statistics(cleaned_df, 'value'))import pandas as pd
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

def normalize_text_column(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Normalize text column by converting to lowercase and stripping whitespace.
    
    Args:
        df: Input DataFrame
        column: Name of column to normalize
    
    Returns:
        DataFrame with normalized column
    """
    df = df.copy()
    df[column] = df[column].astype(str).str.lower().str.strip()
    return df

def fill_missing_values(df: pd.DataFrame, strategy: str = 'mean', columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Fill missing values in DataFrame columns.
    
    Args:
        df: Input DataFrame
        strategy: Method to fill missing values ('mean', 'median', 'mode', 'constant')
        columns: Specific columns to fill, fills all numeric columns if None
    
    Returns:
        DataFrame with filled missing values
    """
    df = df.copy()
    
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    for col in columns:
        if col in df.columns:
            if strategy == 'mean':
                df[col].fillna(df[col].mean(), inplace=True)
            elif strategy == 'median':
                df[col].fillna(df[col].median(), inplace=True)
            elif strategy == 'mode':
                df[col].fillna(df[col].mode()[0], inplace=True)
            elif strategy == 'constant':
                df[col].fillna(0, inplace=True)
    
    return df

def clean_dataframe(df: pd.DataFrame, 
                   text_columns: Optional[List[str]] = None,
                   numeric_strategy: str = 'mean') -> pd.DataFrame:
    """
    Comprehensive data cleaning pipeline.
    
    Args:
        df: Input DataFrame
        text_columns: List of text columns to normalize
        numeric_strategy: Strategy for filling numeric missing values
    
    Returns:
        Cleaned DataFrame
    """
    # Remove duplicates
    df_clean = remove_duplicates(df)
    
    # Normalize text columns
    if text_columns:
        for col in text_columns:
            if col in df_clean.columns:
                df_clean = normalize_text_column(df_clean, col)
    
    # Fill missing values
    df_clean = fill_missing_values(df_clean, strategy=numeric_strategy)
    
    return df_clean

def validate_dataframe(df: pd.DataFrame) -> dict:
    """
    Validate DataFrame and return summary statistics.
    
    Args:
        df: Input DataFrame
    
    Returns:
        Dictionary with validation results
    """
    validation = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': df.isnull().sum().sum(),
        'duplicate_rows': df.duplicated().sum(),
        'column_types': df.dtypes.to_dict(),
        'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
        'text_columns': df.select_dtypes(include=['object']).columns.tolist()
    }
    
    return validation

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'id': [1, 2, 3, 4, 5, 5],
        'name': ['John', 'Jane', 'Bob', 'Alice', 'Eve', 'eve'],
        'age': [25, 30, None, 35, 40, 40],
        'score': [85.5, 92.0, 78.5, None, 88.0, 88.0]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nValidation Results:")
    print(validate_dataframe(df))
    
    cleaned_df = clean_dataframe(df, text_columns=['name'])
    print("\nCleaned DataFrame:")
    print(cleaned_df)
    print("\nCleaned Validation Results:")
    print(validate_dataframe(cleaned_df))
import pandas as pd

def clean_dataframe(df, drop_na=True, rename_columns=True):
    """
    Clean a pandas DataFrame by removing null values and standardizing column names.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        drop_na (bool): If True, drop rows with any null values.
        rename_columns (bool): If True, rename columns to lowercase with underscores.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_na:
        cleaned_df = cleaned_df.dropna()
    
    if rename_columns:
        cleaned_df.columns = (
            cleaned_df.columns
            .str.lower()
            .str.replace(' ', '_')
            .str.replace(r'[^\w_]', '', regex=True)
        )
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and required columns.
    
    Args:
        df (pd.DataFrame): DataFrame to validate.
        required_columns (list): List of required column names.
    
    Returns:
        bool: True if validation passes, False otherwise.
    """
    if not isinstance(df, pd.DataFrame):
        return False
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Missing required columns: {missing_columns}")
            return False
    
    return True

if __name__ == "__main__":
    sample_data = {
        'First Name': ['Alice', 'Bob', None],
        'Last Name': ['Smith', 'Johnson', 'Williams'],
        'Age': [25, 30, 35]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    
    cleaned = clean_dataframe(df)
    print("\nCleaned DataFrame:")
    print(cleaned)
    
    is_valid = validate_dataframe(cleaned, ['first_name', 'last_name'])
    print(f"\nDataFrame validation: {is_valid}")
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
                elif method == 'robust':
                    median = self.df[col].median()
                    iqr = self.df[col].quantile(0.75) - self.df[col].quantile(0.25)
                    if iqr != 0:
                        df_normalized[col] = (self.df[col] - median) / iqr
        
        self.df = df_normalized
        return self.df
    
    def handle_missing_values(self, strategy='mean', columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
            
        df_filled = self.df.copy()
        for col in columns:
            if col in self.df.columns and self.df[col].isnull().any():
                if strategy == 'mean' and pd.api.types.is_numeric_dtype(self.df[col]):
                    df_filled[col] = self.df[col].fillna(self.df[col].mean())
                elif strategy == 'median' and pd.api.types.is_numeric_dtype(self.df[col]):
                    df_filled[col] = self.df[col].fillna(self.df[col].median())
                elif strategy == 'mode':
                    df_filled[col] = self.df[col].fillna(self.df[col].mode()[0])
                elif strategy == 'drop':
                    df_filled = df_filled.dropna(subset=[col])
        
        self.df = df_filled.reset_index(drop=True)
        return self.df
    
    def get_summary(self):
        summary = {
            'original_rows': self.original_shape[0],
            'current_rows': self.df.shape[0],
            'columns': self.df.shape[1],
            'missing_values': self.df.isnull().sum().sum(),
            'numeric_columns': list(self.df.select_dtypes(include=[np.number]).columns),
            'categorical_columns': list(self.df.select_dtypes(include=['object']).columns)
        }
        return summary
    
    def get_cleaned_data(self):
        return self.df.copy()import pandas as pd
import numpy as np

def clean_dataset(df, missing_strategy='mean', outlier_method='iqr', columns=None):
    """
    Clean a pandas DataFrame by handling missing values and outliers.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame.
    missing_strategy (str): Strategy for missing values ('mean', 'median', 'mode', 'drop').
    outlier_method (str): Method for outlier detection ('iqr', 'zscore').
    columns (list): Specific columns to clean. If None, clean all numeric columns.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    df_clean = df.copy()
    
    if columns is None:
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
        columns = numeric_cols
    
    for col in columns:
        if col not in df_clean.columns:
            continue
            
        if missing_strategy != 'drop':
            if missing_strategy == 'mean':
                fill_value = df_clean[col].mean()
            elif missing_strategy == 'median':
                fill_value = df_clean[col].median()
            elif missing_strategy == 'mode':
                fill_value = df_clean[col].mode()[0] if not df_clean[col].mode().empty else 0
            else:
                fill_value = 0
            df_clean[col].fillna(fill_value, inplace=True)
        else:
            df_clean = df_clean.dropna(subset=[col])
        
        if outlier_method == 'iqr':
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df_clean[col] = np.where((df_clean[col] < lower_bound) | (df_clean[col] > upper_bound), 
                                     df_clean[col].median(), df_clean[col])
        elif outlier_method == 'zscore':
            mean_val = df_clean[col].mean()
            std_val = df_clean[col].std()
            if std_val > 0:
                z_scores = np.abs((df_clean[col] - mean_val) / std_val)
                df_clean[col] = np.where(z_scores > 3, mean_val, df_clean[col])
    
    return df_clean

def remove_duplicates(df, subset=None, keep='first'):
    """
    Remove duplicate rows from DataFrame.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame.
    subset (list): Columns to consider for duplicates.
    keep (str): Which duplicates to keep ('first', 'last', False).
    
    Returns:
    pd.DataFrame: DataFrame with duplicates removed.
    """
    return df.drop_duplicates(subset=subset, keep=keep)

def normalize_data(df, columns=None, method='minmax'):
    """
    Normalize specified columns in DataFrame.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame.
    columns (list): Columns to normalize.
    method (str): Normalization method ('minmax', 'standard').
    
    Returns:
    pd.DataFrame: DataFrame with normalized columns.
    """
    df_norm = df.copy()
    
    if columns is None:
        numeric_cols = df_norm.select_dtypes(include=[np.number]).columns.tolist()
        columns = numeric_cols
    
    for col in columns:
        if col not in df_norm.columns:
            continue
            
        if method == 'minmax':
            min_val = df_norm[col].min()
            max_val = df_norm[col].max()
            if max_val > min_val:
                df_norm[col] = (df_norm[col] - min_val) / (max_val - min_val)
        elif method == 'standard':
            mean_val = df_norm[col].mean()
            std_val = df_norm[col].std()
            if std_val > 0:
                df_norm[col] = (df_norm[col] - mean_val) / std_val
    
    return df_norm

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, np.nan, 4, 100],
        'B': [5, 6, 7, 8, 9],
        'C': [10, 11, 12, 13, 14]
    }
    df_sample = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df_sample)
    
    cleaned_df = clean_dataset(df_sample, missing_strategy='median', outlier_method='iqr')
    print("\nCleaned DataFrame:")
    print(cleaned_df)
    
    normalized_df = normalize_data(cleaned_df, method='minmax')
    print("\nNormalized DataFrame:")
    print(normalized_df)
import pandas as pd
import re

def clean_dataframe(df, column_mapping=None, drop_duplicates=True, normalize_text=True):
    """
    Clean a pandas DataFrame by removing duplicates and normalizing text columns.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean
        column_mapping (dict): Optional dictionary to rename columns
        drop_duplicates (bool): Whether to remove duplicate rows
        normalize_text (bool): Whether to normalize text columns
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    
    df_clean = df.copy()
    
    if column_mapping:
        df_clean = df_clean.rename(columns=column_mapping)
    
    if drop_duplicates:
        df_clean = df_clean.drop_duplicates().reset_index(drop=True)
    
    if normalize_text:
        for col in df_clean.select_dtypes(include=['object']).columns:
            df_clean[col] = df_clean[col].apply(_normalize_string)
    
    return df_clean

def _normalize_string(text):
    """
    Normalize a string by converting to lowercase, removing extra whitespace,
    and stripping special characters.
    
    Args:
        text (str): Input string to normalize
    
    Returns:
        str: Normalized string
    """
    if pd.isna(text):
        return text
    
    text = str(text)
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
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
    validation_results['email_valid'] = validation_results[email_column].apply(
        lambda x: bool(re.match(email_pattern, str(x))) if pd.notna(x) else False
    )
    
    return validation_results
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

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate
    required_columns (list): List of required column names
    
    Returns:
    bool: True if validation passes
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    
    if df.empty:
        raise ValueError("DataFrame is empty")
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
    
    return True

if __name__ == "__main__":
    sample_data = {
        'temperature': [22, 23, 24, 25, 26, 100, 27, 28, -10, 29, 30],
        'humidity': [45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55],
        'pressure': [1013, 1012, 1014, 1015, 1016, 2000, 1017, 1018, 500, 1019, 1020]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print(f"\nOriginal shape: {df.shape}")
    
    cleaned_df = clean_numeric_data(df)
    print(f"\nCleaned shape: {cleaned_df.shape}")
    print("\nCleaned DataFrame:")
    print(cleaned_df)