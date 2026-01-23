
import pandas as pd
import numpy as np
from typing import Union, List, Optional

def clean_dataset(
    df: pd.DataFrame,
    drop_duplicates: bool = True,
    handle_missing: str = 'drop',
    numeric_columns: Optional[List[str]] = None,
    categorical_columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Clean a pandas DataFrame by removing duplicates, handling missing values,
    and converting data types.
    """
    df_clean = df.copy()
    
    if drop_duplicates:
        df_clean = df_clean.drop_duplicates()
    
    if handle_missing == 'drop':
        df_clean = df_clean.dropna()
    elif handle_missing == 'fill':
        if numeric_columns:
            df_clean[numeric_columns] = df_clean[numeric_columns].fillna(0)
        if categorical_columns:
            df_clean[categorical_columns] = df_clean[categorical_columns].fillna('Unknown')
    
    if numeric_columns:
        for col in numeric_columns:
            if col in df_clean.columns:
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    
    if categorical_columns:
        for col in categorical_columns:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].astype('category')
    
    return df_clean

def validate_dataframe(df: pd.DataFrame, required_columns: List[str]) -> bool:
    """
    Validate that a DataFrame contains all required columns.
    """
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"Missing columns: {missing_columns}")
        return False
    return True

def calculate_statistics(df: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame:
    """
    Calculate basic statistics for numeric columns.
    """
    stats = df[numeric_cols].agg(['mean', 'median', 'std', 'min', 'max'])
    return stats.T

if __name__ == "__main__":
    sample_data = {
        'id': [1, 2, 2, 3, 4],
        'value': [10.5, 20.3, 20.3, None, 40.7],
        'category': ['A', 'B', 'B', 'C', None]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    
    cleaned_df = clean_dataset(
        df,
        drop_duplicates=True,
        handle_missing='fill',
        numeric_columns=['value'],
        categorical_columns=['category']
    )
    
    print("\nCleaned DataFrame:")
    print(cleaned_df)
    
    if validate_dataframe(cleaned_df, ['id', 'value', 'category']):
        stats = calculate_statistics(cleaned_df, ['value'])
        print("\nStatistics:")
        print(stats)