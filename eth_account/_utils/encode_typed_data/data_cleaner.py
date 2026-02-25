
import pandas as pd
import numpy as np
from typing import List, Optional

def remove_duplicates(
    data: pd.DataFrame,
    subset: Optional[List[str]] = None,
    keep: str = 'first',
    inplace: bool = False
) -> pd.DataFrame:
    """
    Remove duplicate rows from a pandas DataFrame.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Input DataFrame containing potential duplicates
    subset : list, optional
        Column labels to consider for identifying duplicates
    keep : {'first', 'last', False}
        Determines which duplicates to keep
    inplace : bool
        If True, perform operation in-place
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with duplicates removed
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    
    if inplace:
        data.drop_duplicates(subset=subset, keep=keep, inplace=True)
        return data
    else:
        return data.drop_duplicates(subset=subset, keep=keep)

def validate_dataframe(df: pd.DataFrame) -> bool:
    """
    Perform basic validation on DataFrame.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to validate
        
    Returns:
    --------
    bool
        True if DataFrame passes validation checks
    """
    if df.empty:
        print("Warning: DataFrame is empty")
        return False
    
    if df.isnull().all().any():
        print("Warning: Some columns contain only null values")
        return False
    
    return True

def clean_numeric_columns(
    df: pd.DataFrame,
    columns: List[str],
    fill_method: str = 'mean'
) -> pd.DataFrame:
    """
    Clean numeric columns by handling missing values.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    columns : list
        List of numeric column names to clean
    fill_method : {'mean', 'median', 'zero'}
        Method for filling missing values
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with cleaned numeric columns
    """
    df_clean = df.copy()
    
    for col in columns:
        if col not in df_clean.columns:
            print(f"Warning: Column '{col}' not found in DataFrame")
            continue
            
        if not np.issubdtype(df_clean[col].dtype, np.number):
            print(f"Warning: Column '{col}' is not numeric")
            continue
        
        if fill_method == 'mean':
            fill_value = df_clean[col].mean()
        elif fill_method == 'median':
            fill_value = df_clean[col].median()
        elif fill_method == 'zero':
            fill_value = 0
        else:
            raise ValueError("fill_method must be 'mean', 'median', or 'zero'")
        
        df_clean[col] = df_clean[col].fillna(fill_value)
    
    return df_clean

if __name__ == "__main__":
    # Example usage
    sample_data = pd.DataFrame({
        'id': [1, 2, 2, 3, 4],
        'value': [10.5, 20.3, 20.3, None, 40.1],
        'category': ['A', 'B', 'B', 'C', 'A']
    })
    
    print("Original data:")
    print(sample_data)
    print("\nAfter removing duplicates:")
    cleaned = remove_duplicates(sample_data, subset=['id'])
    print(cleaned)
    
    print("\nAfter cleaning numeric columns:")
    final = clean_numeric_columns(cleaned, columns=['value'])
    print(final)
import numpy as np

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a specified column using the IQR method.
    
    Parameters:
    data (pd.DataFrame): The input DataFrame.
    column (str): The column name to process.
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed.
    """
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data