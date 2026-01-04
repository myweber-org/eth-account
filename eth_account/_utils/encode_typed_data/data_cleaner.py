
import pandas as pd
import numpy as np
from typing import List, Optional

class DataCleaner:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.original_shape = df.shape

    def remove_duplicates(self, subset: Optional[List[str]] = None) -> 'DataCleaner':
        self.df = self.df.drop_duplicates(subset=subset, keep='first')
        return self

    def handle_missing_values(self, strategy: str = 'drop', fill_value: Optional[float] = None) -> 'DataCleaner':
        if strategy == 'drop':
            self.df = self.df.dropna()
        elif strategy == 'fill':
            if fill_value is not None:
                self.df = self.df.fillna(fill_value)
            else:
                numeric_cols = self.df.select_dtypes(include=[np.number]).columns
                self.df[numeric_cols] = self.df[numeric_cols].fillna(self.df[numeric_cols].mean())
        return self

    def normalize_column_names(self) -> 'DataCleaner':
        self.df.columns = [col.strip().lower().replace(' ', '_') for col in self.df.columns]
        return self

    def get_cleaned_data(self) -> pd.DataFrame:
        return self.df

    def get_cleaning_report(self) -> dict:
        return {
            'original_rows': self.original_shape[0],
            'cleaned_rows': self.df.shape[0],
            'original_columns': self.original_shape[1],
            'cleaned_columns': self.df.shape[1],
            'rows_removed': self.original_shape[0] - self.df.shape[0],
            'columns_removed': self.original_shape[1] - self.df.shape[1]
        }

def clean_dataset(df: pd.DataFrame, steps: List[str] = None) -> pd.DataFrame:
    if steps is None:
        steps = ['normalize', 'deduplicate', 'handle_missing']
    
    cleaner = DataCleaner(df)
    
    if 'normalize' in steps:
        cleaner.normalize_column_names()
    
    if 'deduplicate' in steps:
        cleaner.remove_duplicates()
    
    if 'handle_missing' in steps:
        cleaner.handle_missing_values(strategy='fill')
    
    return cleaner.get_cleaned_data()
import pandas as pd

def remove_duplicates(df, subset=None, keep='first'):
    """
    Remove duplicate rows from a DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        subset (list, optional): Column names to consider for duplicates
        keep (str, optional): Which duplicates to keep ('first', 'last', False)
    
    Returns:
        pd.DataFrame: DataFrame with duplicates removed
    """
    if df.empty:
        return df
    
    cleaned_df = df.drop_duplicates(subset=subset, keep=keep)
    
    removed_count = len(df) - len(cleaned_df)
    if removed_count > 0:
        print(f"Removed {removed_count} duplicate row(s)")
    
    return cleaned_df

def clean_numeric_columns(df, columns):
    """
    Clean numeric columns by converting to appropriate types and handling errors.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        columns (list): List of column names to clean
    
    Returns:
        pd.DataFrame: DataFrame with cleaned numeric columns
    """
    cleaned_df = df.copy()
    
    for col in columns:
        if col in cleaned_df.columns:
            cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='coerce')
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        required_columns (list, optional): List of required column names
    
    Returns:
        tuple: (is_valid, error_message)
    """
    if df.empty:
        return False, "DataFrame is empty"
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
    
    return True, "DataFrame is valid"
import pandas as pd
import re

def clean_dataframe(df, column_mapping=None, drop_duplicates=True, normalize_text=True):
    """
    Clean a pandas DataFrame by removing duplicates and normalizing text columns.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        column_mapping (dict, optional): Dictionary mapping old column names to new ones.
        drop_duplicates (bool): Whether to remove duplicate rows.
        normalize_text (bool): Whether to normalize text columns by stripping whitespace and lowercasing.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if column_mapping:
        cleaned_df = cleaned_df.rename(columns=column_mapping)
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates().reset_index(drop=True)
    
    if normalize_text:
        for col in cleaned_df.select_dtypes(include=['object']).columns:
            cleaned_df[col] = cleaned_df[col].astype(str).apply(
                lambda x: re.sub(r'\s+', ' ', x.strip()).lower() if pd.notnull(x) else x
            )
    
    return cleaned_df

def validate_email(email):
    """
    Validate email format using regex.
    
    Args:
        email (str): Email address to validate.
    
    Returns:
        bool: True if email format is valid, False otherwise.
    """
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, str(email)))

def sample_data():
    """
    Generate sample data for testing.
    
    Returns:
        pd.DataFrame: Sample DataFrame with mixed data.
    """
    data = {
        'name': ['John Doe', 'Jane Smith', 'John Doe', 'Bob Johnson', 'Alice Brown'],
        'email': ['john@example.com', 'jane@test.org', 'invalid-email', 'bob@company.net', 'alice@domain.co'],
        'age': [25, 30, 25, 35, 28],
        'city': ['New York', 'Los Angeles', 'New York', 'Chicago', 'Boston']
    }
    return pd.DataFrame(data)

if __name__ == '__main__':
    df = sample_data()
    print("Original DataFrame:")
    print(df)
    print("\nCleaned DataFrame:")
    cleaned = clean_dataframe(df, column_mapping={'city': 'location'})
    print(cleaned)
    print("\nValid emails:")
    df['valid_email'] = df['email'].apply(validate_email)
    print(df[['email', 'valid_email']])