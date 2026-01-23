
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
        print(stats)import pandas as pd
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
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        removed = initial_rows - len(cleaned_df)
        print(f"Removed {removed} duplicate rows.")
    
    if fill_missing == 'drop':
        cleaned_df = cleaned_df.dropna()
        print("Dropped rows with missing values.")
    elif fill_missing in ['mean', 'median']:
        numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if fill_missing == 'mean':
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mean())
            else:
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
        print(f"Filled missing numeric values with {fill_missing}.")
    elif fill_missing == 'mode':
        for col in cleaned_df.columns:
            cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mode()[0] if not cleaned_df[col].mode().empty else None)
        print("Filled missing values with mode.")
    
    return cleaned_df

def validate_data(df, required_columns=None, min_rows=1):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of required column names.
    min_rows (int): Minimum number of rows required.
    
    Returns:
    bool: True if validation passes, False otherwise.
    """
    if df.empty:
        print("Validation failed: DataFrame is empty.")
        return False
    
    if len(df) < min_rows:
        print(f"Validation failed: DataFrame has fewer than {min_rows} rows.")
        return False
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            print(f"Validation failed: Missing required columns: {missing_cols}")
            return False
    
    print("Data validation passed.")
    return True

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, 2, 3, None, 5],
        'B': [10, None, 30, 40, 50, 60],
        'C': ['x', 'y', 'y', 'z', None, 'x']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    
    cleaned = clean_dataset(df, fill_missing='mean')
    print("\nCleaned DataFrame:")
    print(cleaned)
    
    is_valid = validate_data(cleaned, required_columns=['A', 'B'], min_rows=3)
    print(f"\nData valid: {is_valid}")