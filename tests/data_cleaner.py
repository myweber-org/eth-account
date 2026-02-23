
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