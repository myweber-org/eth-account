
import pandas as pd
import numpy as np
from typing import List, Optional

def remove_duplicate_rows(
    data: pd.DataFrame,
    subset: Optional[List[str]] = None,
    keep: str = 'first',
    reset_index: bool = True
) -> pd.DataFrame:
    """
    Remove duplicate rows from a pandas DataFrame.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Input DataFrame containing potential duplicates
    subset : list, optional
        Column names to consider for identifying duplicates
    keep : str, default 'first'
        Which duplicates to keep: 'first', 'last', or False
    reset_index : bool, default True
        Whether to reset the DataFrame index after removal
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with duplicates removed
    """
    if data.empty:
        return data
    
    cleaned_data = data.drop_duplicates(subset=subset, keep=keep)
    
    if reset_index:
        cleaned_data = cleaned_data.reset_index(drop=True)
    
    return cleaned_data

def find_duplicate_indices(
    data: pd.DataFrame,
    subset: Optional[List[str]] = None
) -> pd.Index:
    """
    Find indices of duplicate rows in a DataFrame.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Input DataFrame to check for duplicates
    subset : list, optional
        Column names to consider for identifying duplicates
    
    Returns:
    --------
    pd.Index
        Indices of duplicate rows
    """
    if data.empty:
        return pd.Index([])
    
    duplicates = data.duplicated(subset=subset, keep=False)
    return data.index[duplicates]

def clean_numeric_outliers(
    data: pd.DataFrame,
    column: str,
    method: str = 'iqr',
    threshold: float = 1.5
) -> pd.DataFrame:
    """
    Remove outliers from a numeric column using specified method.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Input DataFrame
    column : str
        Column name to clean
    method : str, default 'iqr'
        Outlier detection method: 'iqr' or 'zscore'
    threshold : float, default 1.5
        Threshold multiplier for IQR or Z-score cutoff
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with outliers removed from specified column
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    numeric_data = pd.to_numeric(data[column], errors='coerce')
    
    if method == 'iqr':
        q1 = numeric_data.quantile(0.25)
        q3 = numeric_data.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr
        mask = (numeric_data >= lower_bound) & (numeric_data <= upper_bound)
    elif method == 'zscore':
        mean = numeric_data.mean()
        std = numeric_data.std()
        z_scores = np.abs((numeric_data - mean) / std)
        mask = z_scores <= threshold
    else:
        raise ValueError("Method must be 'iqr' or 'zscore'")
    
    return data[mask].reset_index(drop=True)

def validate_dataframe(
    data: pd.DataFrame,
    required_columns: List[str],
    allow_nan: bool = False
) -> bool:
    """
    Validate DataFrame structure and content.
    
    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame to validate
    required_columns : list
        List of column names that must be present
    allow_nan : bool, default False
        Whether to allow NaN values in the DataFrame
    
    Returns:
    --------
    bool
        True if DataFrame passes validation, False otherwise
    """
    if not isinstance(data, pd.DataFrame):
        return False
    
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        print(f"Missing required columns: {missing_columns}")
        return False
    
    if not allow_nan and data.isnull().any().any():
        print("DataFrame contains NaN values")
        return False
    
    return True

def main():
    """Example usage of data cleaning functions."""
    sample_data = pd.DataFrame({
        'id': [1, 2, 3, 2, 4, 1],
        'name': ['Alice', 'Bob', 'Charlie', 'Bob', 'David', 'Alice'],
        'score': [85, 92, 78, 92, 150, 85],
        'department': ['HR', 'IT', 'IT', 'IT', 'Sales', 'HR']
    })
    
    print("Original data:")
    print(sample_data)
    print("\n" + "="*50 + "\n")
    
    cleaned = remove_duplicate_rows(sample_data, subset=['id', 'name'])
    print("After removing duplicates:")
    print(cleaned)
    print("\n" + "="*50 + "\n")
    
    duplicate_indices = find_duplicate_indices(sample_data, subset=['name'])
    print(f"Duplicate indices: {list(duplicate_indices)}")
    print("\n" + "="*50 + "\n")
    
    outlier_cleaned = clean_numeric_outliers(sample_data, 'score', method='iqr')
    print("After removing score outliers:")
    print(outlier_cleaned)
    print("\n" + "="*50 + "\n")
    
    is_valid = validate_dataframe(sample_data, required_columns=['id', 'name', 'score'])
    print(f"DataFrame validation: {is_valid}")

if __name__ == "__main__":
    main()