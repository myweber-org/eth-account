
import pandas as pd
import numpy as np
from typing import Optional

def clean_csv_data(
    input_path: str,
    output_path: str,
    missing_strategy: str = 'mean',
    columns_to_drop: Optional[list] = None
) -> pd.DataFrame:
    """
    Load and clean CSV data by handling missing values and dropping specified columns.
    
    Parameters:
    input_path: Path to input CSV file
    output_path: Path to save cleaned CSV file
    missing_strategy: Strategy for handling missing values ('mean', 'median', 'mode', 'drop')
    columns_to_drop: List of column names to remove from dataset
    
    Returns:
    Cleaned pandas DataFrame
    """
    
    df = pd.read_csv(input_path)
    
    if columns_to_drop:
        df = df.drop(columns=columns_to_drop, errors='ignore')
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        if df[col].isnull().any():
            if missing_strategy == 'mean':
                fill_value = df[col].mean()
            elif missing_strategy == 'median':
                fill_value = df[col].median()
            elif missing_strategy == 'mode':
                fill_value = df[col].mode()[0]
            elif missing_strategy == 'drop':
                df = df.dropna(subset=[col])
                continue
            else:
                raise ValueError(f"Unknown strategy: {missing_strategy}")
            
            df[col] = df[col].fillna(fill_value)
    
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna('Unknown')
    
    df.to_csv(output_path, index=False)
    
    print(f"Data cleaning completed. Cleaned data saved to: {output_path}")
    print(f"Original shape: {pd.read_csv(input_path).shape}")
    print(f"Cleaned shape: {df.shape}")
    
    return df

def validate_dataframe(df: pd.DataFrame) -> bool:
    """
    Validate that DataFrame has no null values and positive numeric columns.
    
    Parameters:
    df: DataFrame to validate
    
    Returns:
    Boolean indicating if data is valid
    """
    
    if df.isnull().any().any():
        print("Validation failed: DataFrame contains null values")
        return False
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if (df[col] < 0).any():
            print(f"Validation warning: Column '{col}' contains negative values")
    
    return True

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'A': [1, 2, np.nan, 4, 5],
        'B': [10.5, np.nan, 30.2, 40.1, 50.0],
        'C': ['X', 'Y', 'Z', np.nan, 'W'],
        'D': [100, 200, 300, 400, 500]
    })
    
    sample_data.to_csv('sample_data.csv', index=False)
    
    cleaned_df = clean_csv_data(
        input_path='sample_data.csv',
        output_path='cleaned_data.csv',
        missing_strategy='mean',
        columns_to_drop=['D']
    )
    
    is_valid = validate_dataframe(cleaned_df)
    print(f"Data validation result: {is_valid}")
import pandas as pd
import numpy as np

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.numeric_columns = self.df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_columns = self.df.select_dtypes(include=['object']).columns.tolist()
    
    def handle_missing_values(self, strategy='mean', fill_value=None):
        if strategy == 'mean' and self.numeric_columns:
            self.df[self.numeric_columns] = self.df[self.numeric_columns].fillna(self.df[self.numeric_columns].mean())
        elif strategy == 'median' and self.numeric_columns:
            self.df[self.numeric_columns] = self.df[self.numeric_columns].fillna(self.df[self.numeric_columns].median())
        elif strategy == 'mode' and self.categorical_columns:
            for col in self.categorical_columns:
                self.df[col] = self.df[col].fillna(self.df[col].mode()[0] if not self.df[col].mode().empty else 'Unknown')
        elif fill_value is not None:
            self.df = self.df.fillna(fill_value)
        return self
    
    def remove_outliers_iqr(self, columns=None, multiplier=1.5):
        if columns is None:
            columns = self.numeric_columns
        
        for col in columns:
            if col in self.numeric_columns:
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - multiplier * IQR
                upper_bound = Q3 + multiplier * IQR
                self.df = self.df[(self.df[col] >= lower_bound) & (self.df[col] <= upper_bound)]
        return self
    
    def standardize_data(self, columns=None):
        if columns is None:
            columns = self.numeric_columns
        
        for col in columns:
            if col in self.numeric_columns:
                mean = self.df[col].mean()
                std = self.df[col].std()
                if std > 0:
                    self.df[col] = (self.df[col] - mean) / std
        return self
    
    def normalize_data(self, columns=None):
        if columns is None:
            columns = self.numeric_columns
        
        for col in columns:
            if col in self.numeric_columns:
                min_val = self.df[col].min()
                max_val = self.df[col].max()
                if max_val > min_val:
                    self.df[col] = (self.df[col] - min_val) / (max_val - min_val)
        return self
    
    def get_cleaned_data(self):
        return self.df
    
    def get_summary(self):
        summary = {
            'original_shape': self.df.shape,
            'missing_values': self.df.isnull().sum().sum(),
            'numeric_columns': len(self.numeric_columns),
            'categorical_columns': len(self.categorical_columns)
        }
        return summary

def clean_dataset(df, missing_strategy='mean', remove_outliers=True):
    cleaner = DataCleaner(df)
    cleaner.handle_missing_values(strategy=missing_strategy)
    
    if remove_outliers:
        cleaner.remove_outliers_iqr()
    
    cleaned_df = cleaner.get_cleaned_data()
    return cleaned_df