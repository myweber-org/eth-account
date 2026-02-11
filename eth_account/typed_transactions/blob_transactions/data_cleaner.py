import pandas as pd
import numpy as np

def clean_dataset(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
        print(f"Removed {len(df) - len(cleaned_df)} duplicate rows.")
    
    if cleaned_df.isnull().sum().any():
        print("Handling missing values...")
        numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
        
        if fill_missing == 'mean':
            for col in numeric_cols:
                if cleaned_df[col].isnull().any():
                    cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mean())
        elif fill_missing == 'median':
            for col in numeric_cols:
                if cleaned_df[col].isnull().any():
                    cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
        elif fill_missing == 'zero':
            cleaned_df[numeric_cols] = cleaned_df[numeric_cols].fillna(0)
        else:
            cleaned_df = cleaned_df.dropna()
            print("Dropped rows with missing values.")
    
    print(f"Original shape: {df.shape}")
    print(f"Cleaned shape: {cleaned_df.shape}")
    
    return cleaned_df

def validate_data(df, required_columns=None):
    """
    Validate the DataFrame structure and content.
    """
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
    
    if df.empty:
        raise ValueError("DataFrame is empty")
    
    return True
def remove_duplicates_preserve_order(sequence):
    seen = set()
    result = []
    for item in sequence:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result
import pandas as pd
import numpy as np

def clean_csv_data(file_path, fill_strategy='mean'):
    """
    Load a CSV file, handle missing values, and return cleaned DataFrame.
    
    Parameters:
    file_path (str): Path to the CSV file.
    fill_strategy (str): Strategy for filling missing values. 
                         Options: 'mean', 'median', 'mode', 'zero'.
    
    Returns:
    pandas.DataFrame: Cleaned DataFrame.
    """
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found at path: {file_path}")
    except pd.errors.EmptyDataError:
        raise ValueError("The provided CSV file is empty.")
    
    original_shape = df.shape
    print(f"Original data shape: {original_shape}")
    
    missing_counts = df.isnull().sum()
    if missing_counts.any():
        print("Missing values per column:")
        for col, count in missing_counts[missing_counts > 0].items():
            print(f"  {col}: {count} missing")
        
        if fill_strategy == 'mean':
            df = df.fillna(df.select_dtypes(include=[np.number]).mean())
        elif fill_strategy == 'median':
            df = df.fillna(df.select_dtypes(include=[np.number]).median())
        elif fill_strategy == 'mode':
            df = df.fillna(df.mode().iloc[0])
        elif fill_strategy == 'zero':
            df = df.fillna(0)
        else:
            raise ValueError(f"Unsupported fill strategy: {fill_strategy}")
        
        print(f"Missing values filled using '{fill_strategy}' strategy.")
    else:
        print("No missing values found.")
    
    cleaned_shape = df.shape
    print(f"Cleaned data shape: {cleaned_shape}")
    
    if original_shape != cleaned_shape:
        print("Warning: Data shape changed during cleaning.")
    
    return df

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a specific column using IQR method.
    
    Parameters:
    df (pandas.DataFrame): Input DataFrame.
    column (str): Column name to process.
    
    Returns:
    pandas.DataFrame: DataFrame with outliers removed.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame.")
    
    if not np.issubdtype(df[column].dtype, np.number):
        raise ValueError(f"Column '{column}' must be numeric.")
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    original_len = len(df)
    df_clean = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    removed_count = original_len - len(df_clean)
    
    if removed_count > 0:
        print(f"Removed {removed_count} outliers from column '{column}'.")
    
    return df_clean

def save_cleaned_data(df, output_path):
    """
    Save cleaned DataFrame to CSV file.
    
    Parameters:
    df (pandas.DataFrame): DataFrame to save.
    output_path (str): Path for output CSV file.
    """
    try:
        df.to_csv(output_path, index=False)
        print(f"Cleaned data saved to: {output_path}")
    except Exception as e:
        raise IOError(f"Failed to save data: {str(e)}")

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'A': [1, 2, np.nan, 4, 5],
        'B': [10, np.nan, 30, 40, 50],
        'C': [100, 200, 300, np.nan, 500]
    })
    
    sample_data.to_csv('sample_data.csv', index=False)
    
    cleaned_df = clean_csv_data('sample_data.csv', fill_strategy='mean')
    print("\nCleaned DataFrame:")
    print(cleaned_df)
    
    import os
    if os.path.exists('sample_data.csv'):
        os.remove('sample_data.csv')
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
        Column names to consider for identifying duplicates
    keep : {'first', 'last', False}
        Determines which duplicates to keep
    inplace : bool
        If True, modifies the DataFrame in place
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with duplicates removed
    """
    if not inplace:
        data = data.copy()
    
    if subset is None:
        subset = data.columns.tolist()
    
    # Remove duplicates
    cleaned_data = data.drop_duplicates(subset=subset, keep=keep, inplace=False)
    
    # Log removal statistics
    original_count = len(data)
    cleaned_count = len(cleaned_data)
    duplicates_removed = original_count - cleaned_count
    
    print(f"Original records: {original_count}")
    print(f"Cleaned records: {cleaned_count}")
    print(f"Duplicates removed: {duplicates_removed}")
    
    return cleaned_data

def validate_dataframe(data: pd.DataFrame) -> bool:
    """
    Validate basic DataFrame structure and content.
    
    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame to validate
    
    Returns:
    --------
    bool
        True if DataFrame passes validation checks
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    
    if data.empty:
        print("Warning: DataFrame is empty")
        return False
    
    if data.isnull().all().any():
        print("Warning: Some columns contain only null values")
    
    return True

def clean_numeric_columns(
    data: pd.DataFrame,
    columns: List[str],
    fill_method: str = 'mean'
) -> pd.DataFrame:
    """
    Clean numeric columns by handling missing values.
    
    Parameters:
    -----------
    data : pd.DataFrame
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
    cleaned_data = data.copy()
    
    for col in columns:
        if col not in cleaned_data.columns:
            print(f"Warning: Column '{col}' not found in DataFrame")
            continue
        
        if not pd.api.types.is_numeric_dtype(cleaned_data[col]):
            print(f"Warning: Column '{col}' is not numeric")
            continue
        
        # Count missing values
        missing_count = cleaned_data[col].isnull().sum()
        if missing_count > 0:
            print(f"Column '{col}': {missing_count} missing values")
            
            # Fill missing values based on specified method
            if fill_method == 'mean':
                fill_value = cleaned_data[col].mean()
            elif fill_method == 'median':
                fill_value = cleaned_data[col].median()
            elif fill_method == 'zero':
                fill_value = 0
            else:
                raise ValueError(f"Unknown fill_method: {fill_method}")
            
            cleaned_data[col] = cleaned_data[col].fillna(fill_value)
            print(f"  Filled with {fill_method}: {fill_value}")
    
    return cleaned_data

# Example usage (commented out for production)
# if __name__ == "__main__":
#     # Create sample data
#     sample_data = pd.DataFrame({
#         'id': [1, 2, 2, 3, 4, 4, 5],
#         'value': [10, 20, 20, 30, np.nan, 40, 50],
#         'category': ['A', 'B', 'B', 'C', 'D', 'D', 'E']
#     })
#     
#     # Clean the data
#     cleaned = remove_duplicates(sample_data, subset=['id'])
#     cleaned = clean_numeric_columns(cleaned, columns=['value'])
#     print("\nCleaned DataFrame:")
#     print(cleaned)
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range (IQR) method.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    column (str): The column name to clean.
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed.
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return filtered_df

def calculate_basic_stats(df, column):
    """
    Calculate basic statistics for a column.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    column (str): The column name.
    
    Returns:
    dict: Dictionary containing count, mean, std, min, max.
    """
    stats = {
        'count': df[column].count(),
        'mean': df[column].mean(),
        'std': df[column].std(),
        'min': df[column].min(),
        'max': df[column].max()
    }
    return stats

if __name__ == "__main__":
    sample_data = {'values': [10, 12, 12, 13, 12, 11, 14, 13, 15, 102, 12, 14, 13, 12, 11, 10, 9, 8, 12, 11]}
    df = pd.DataFrame(sample_data)
    
    print("Original DataFrame:")
    print(df)
    print("\nOriginal Statistics:")
    print(calculate_basic_stats(df, 'values'))
    
    cleaned_df = remove_outliers_iqr(df, 'values')
    print("\nCleaned DataFrame:")
    print(cleaned_df)
    print("\nCleaned Statistics:")
    print(calculate_basic_stats(cleaned_df, 'values'))
import numpy as np
import pandas as pd

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return filtered_df

def clean_dataset(df, numeric_columns):
    cleaned_df = df.copy()
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
    cleaned_df = cleaned_df.dropna()
    cleaned_df = cleaned_df.reset_index(drop=True)
    return cleaned_df

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'A': np.random.normal(100, 15, 1000),
        'B': np.random.exponential(50, 1000),
        'C': np.random.uniform(0, 200, 1000)
    })
    cleaned_data = clean_dataset(sample_data, ['A', 'B', 'C'])
    print(f"Original shape: {sample_data.shape}")
    print(f"Cleaned shape: {cleaned_data.shape}")
import pandas as pd
import re

def clean_dataframe(df, column_mapping=None, drop_duplicates=True, normalize_text=True):
    """
    Clean a pandas DataFrame by removing duplicates and normalizing text columns.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        column_mapping (dict, optional): Dictionary mapping original column names to new names.
        drop_duplicates (bool): Whether to remove duplicate rows.
        normalize_text (bool): Whether to normalize text columns (strip, lower case, remove extra spaces).
    
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
                lambda x: re.sub(r'\s+', ' ', x.strip().lower())
            )
    
    return cleaned_df

def validate_email(email):
    """
    Validate email format using regex.
    
    Args:
        email (str): Email address to validate.
    
    Returns:
        bool: True if email is valid, False otherwise.
    """
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))

def filter_valid_emails(df, email_column):
    """
    Filter DataFrame to only include rows with valid email addresses.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        email_column (str): Name of the column containing email addresses.
    
    Returns:
        pd.DataFrame: DataFrame with only valid emails.
    """
    valid_mask = df[email_column].apply(validate_email)
    return df[valid_mask].reset_index(drop=True)