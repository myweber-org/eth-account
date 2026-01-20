
import pandas as pd
import numpy as np

def clean_dataset(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean
        drop_duplicates (bool): Whether to drop duplicate rows
        fill_missing (str): Method to fill missing values ('mean', 'median', 'mode', or 'drop')
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        removed = initial_rows - len(cleaned_df)
        print(f"Removed {removed} duplicate rows")
    
    if cleaned_df.isnull().sum().sum() > 0:
        print(f"Found {cleaned_df.isnull().sum().sum()} missing values")
        
        if fill_missing == 'drop':
            cleaned_df = cleaned_df.dropna()
            print("Dropped rows with missing values")
        elif fill_missing in ['mean', 'median']:
            numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if fill_missing == 'mean':
                    cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mean())
                elif fill_missing == 'median':
                    cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
            print(f"Filled missing numeric values with {fill_missing}")
        elif fill_missing == 'mode':
            for col in cleaned_df.columns:
                if cleaned_df[col].dtype == 'object':
                    cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mode()[0] if not cleaned_df[col].mode().empty else 'Unknown')
            print("Filled missing categorical values with mode")
    
    print(f"Cleaned dataset: {len(cleaned_df)} rows, {len(cleaned_df.columns)} columns")
    return cleaned_df

def validate_dataset(df, required_columns=None, min_rows=1):
    """
    Validate dataset structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        required_columns (list): List of required column names
        min_rows (int): Minimum number of rows required
    
    Returns:
        bool: True if validation passes, False otherwise
    """
    if len(df) < min_rows:
        print(f"Validation failed: Dataset has only {len(df)} rows, minimum required is {min_rows}")
        return False
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            print(f"Validation failed: Missing required columns: {missing_cols}")
            return False
    
    print("Dataset validation passed")
    return True

if __name__ == "__main__":
    sample_data = {
        'id': [1, 2, 2, 3, 4, 5],
        'value': [10.5, 20.3, 20.3, None, 40.1, 50.0],
        'category': ['A', 'B', 'B', 'C', None, 'A']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original dataset:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    cleaned = clean_dataset(df, drop_duplicates=True, fill_missing='mean')
    print("\nCleaned dataset:")
    print(cleaned)
    
    is_valid = validate_dataset(cleaned, required_columns=['id', 'value'], min_rows=3)
    print(f"\nDataset is valid: {is_valid}")import pandas as pd

def clean_dataframe(df):
    """
    Remove duplicate rows and fill missing values with column mean.
    """
    # Remove duplicates
    df_cleaned = df.drop_duplicates()
    
    # Fill missing numeric values with column mean
    numeric_cols = df_cleaned.select_dtypes(include=['number']).columns
    df_cleaned[numeric_cols] = df_cleaned[numeric_cols].fillna(df_cleaned[numeric_cols].mean())
    
    # Fill missing categorical values with mode
    categorical_cols = df_cleaned.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df_cleaned[col].isnull().any():
            mode_value = df_cleaned[col].mode()[0]
            df_cleaned[col] = df_cleaned[col].fillna(mode_value)
    
    return df_cleaned

def validate_data(df):
    """
    Check if dataframe has any remaining missing values.
    """
    missing_values = df.isnull().sum().sum()
    if missing_values == 0:
        return True
    else:
        print(f"Data still contains {missing_values} missing values.")
        return False

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'A': [1, 2, 2, 4, None],
        'B': [5, None, 7, 8, 9],
        'C': ['x', 'y', 'y', None, 'z']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    
    cleaned_df = clean_dataframe(df)
    print("\nCleaned DataFrame:")
    print(cleaned_df)
    
    is_valid = validate_data(cleaned_df)
    print(f"\nData validation passed: {is_valid}")import pandas as pd
import numpy as np
from typing import Union, List, Dict, Any

def remove_duplicates(df: pd.DataFrame, subset: List[str] = None) -> pd.DataFrame:
    """
    Remove duplicate rows from DataFrame.
    
    Args:
        df: Input DataFrame
        subset: Columns to consider for duplicates
    
    Returns:
        DataFrame with duplicates removed
    """
    return df.drop_duplicates(subset=subset, keep='first')

def convert_column_types(df: pd.DataFrame, 
                         type_map: Dict[str, str]) -> pd.DataFrame:
    """
    Convert columns to specified data types.
    
    Args:
        df: Input DataFrame
        type_map: Dictionary mapping column names to target types
    
    Returns:
        DataFrame with converted types
    """
    df_copy = df.copy()
    
    for column, dtype in type_map.items():
        if column in df_copy.columns:
            try:
                if dtype == 'datetime':
                    df_copy[column] = pd.to_datetime(df_copy[column])
                elif dtype == 'numeric':
                    df_copy[column] = pd.to_numeric(df_copy[column], errors='coerce')
                elif dtype == 'category':
                    df_copy[column] = df_copy[column].astype('category')
                else:
                    df_copy[column] = df_copy[column].astype(dtype)
            except Exception as e:
                print(f"Error converting column {column}: {e}")
    
    return df_copy

def handle_missing_values(df: pd.DataFrame, 
                         strategy: str = 'drop',
                         fill_value: Any = None) -> pd.DataFrame:
    """
    Handle missing values in DataFrame.
    
    Args:
        df: Input DataFrame
        strategy: 'drop', 'fill', or 'interpolate'
        fill_value: Value to use when filling
    
    Returns:
        DataFrame with handled missing values
    """
    if strategy == 'drop':
        return df.dropna()
    elif strategy == 'fill':
        return df.fillna(fill_value)
    elif strategy == 'interpolate':
        return df.interpolate()
    else:
        raise ValueError("Strategy must be 'drop', 'fill', or 'interpolate'")

def clean_dataframe(df: pd.DataFrame,
                   deduplicate: bool = True,
                   type_conversions: Dict[str, str] = None,
                   missing_strategy: str = 'drop') -> pd.DataFrame:
    """
    Apply multiple cleaning operations to DataFrame.
    
    Args:
        df: Input DataFrame
        deduplicate: Whether to remove duplicates
        type_conversions: Type conversion mapping
        missing_strategy: Strategy for handling missing values
    
    Returns:
        Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    if deduplicate:
        cleaned_df = remove_duplicates(cleaned_df)
    
    if type_conversions:
        cleaned_df = convert_column_types(cleaned_df, type_conversions)
    
    cleaned_df = handle_missing_values(cleaned_df, strategy=missing_strategy)
    
    return cleaned_df

def validate_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validate DataFrame and return statistics.
    
    Args:
        df: Input DataFrame
    
    Returns:
        Dictionary with validation statistics
    """
    stats = {
        'row_count': len(df),
        'column_count': len(df.columns),
        'missing_values': df.isnull().sum().sum(),
        'duplicate_rows': df.duplicated().sum(),
        'data_types': df.dtypes.to_dict(),
        'memory_usage': df.memory_usage(deep=True).sum()
    }
    
    return stats

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'id': [1, 2, 2, 3, 4],
        'name': ['Alice', 'Bob', 'Bob', 'Charlie', None],
        'age': ['25', '30', '30', '35', '40'],
        'score': [85.5, 92.0, 92.0, None, 88.5]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nValidation stats:")
    print(validate_dataframe(df))
    
    cleaned = clean_dataframe(
        df,
        type_conversions={'age': 'numeric', 'score': 'numeric'},
        missing_strategy='fill'
    )
    
    print("\nCleaned DataFrame:")
    print(cleaned)
    print("\nCleaned validation stats:")
    print(validate_dataframe(cleaned))