
import pandas as pd

def clean_column_data(df, column_name):
    """
    Clean a specified column in a DataFrame by stripping whitespace and converting to lowercase.
    
    Parameters:
    df (pd.DataFrame): The DataFrame containing the column to clean.
    column_name (str): The name of the column to clean.
    
    Returns:
    pd.DataFrame: A DataFrame with the cleaned column.
    
    Raises:
    ValueError: If the specified column does not exist in the DataFrame.
    """
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame.")
    
    df[column_name] = df[column_name].astype(str).str.strip().str.lower()
    return df
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

    def normalize_column(self, column_name: str) -> 'DataCleaner':
        if column_name in self.df.columns:
            col_data = self.df[column_name]
            if pd.api.types.is_numeric_dtype(col_data):
                mean_val = col_data.mean()
                std_val = col_data.std()
                if std_val > 0:
                    self.df[column_name] = (col_data - mean_val) / std_val
        return self

    def fill_missing_values(self, strategy: str = 'mean') -> 'DataCleaner':
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if self.df[col].isnull().any():
                if strategy == 'mean':
                    fill_value = self.df[col].mean()
                elif strategy == 'median':
                    fill_value = self.df[col].median()
                elif strategy == 'zero':
                    fill_value = 0
                else:
                    continue
                self.df[col].fillna(fill_value, inplace=True)
        return self

    def get_cleaned_data(self) -> pd.DataFrame:
        return self.df

    def get_cleaning_report(self) -> dict:
        removed_rows = self.original_shape[0] - self.df.shape[0]
        removed_cols = self.original_shape[1] - self.df.shape[1]
        return {
            'original_shape': self.original_shape,
            'cleaned_shape': self.df.shape,
            'removed_rows': removed_rows,
            'removed_cols': removed_cols,
            'remaining_missing': self.df.isnull().sum().sum()
        }

def clean_dataset(dataframe: pd.DataFrame, 
                  remove_dups: bool = True,
                  normalize_cols: Optional[List[str]] = None,
                  fill_missing: bool = True) -> pd.DataFrame:
    cleaner = DataCleaner(dataframe)
    
    if remove_dups:
        cleaner.remove_duplicates()
    
    if normalize_cols:
        for col in normalize_cols:
            cleaner.normalize_column(col)
    
    if fill_missing:
        cleaner.fill_missing_values()
    
    return cleaner.get_cleaned_data()