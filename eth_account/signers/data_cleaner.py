
import pandas as pd

def remove_duplicates(df, subset=None, keep='first'):
    """
    Remove duplicate rows from a DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        subset (list, optional): Columns to consider for duplicates
        keep (str, optional): Which duplicates to keep ('first', 'last', False)
    
    Returns:
        pd.DataFrame: DataFrame with duplicates removed
    """
    if subset is None:
        subset = df.columns.tolist()
    
    cleaned_df = df.drop_duplicates(subset=subset, keep=keep)
    
    removed_count = len(df) - len(cleaned_df)
    print(f"Removed {removed_count} duplicate rows")
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        required_columns (list, optional): List of required columns
    
    Returns:
        bool: True if validation passes, False otherwise
    """
    if not isinstance(df, pd.DataFrame):
        print("Error: Input is not a pandas DataFrame")
        return False
    
    if df.empty:
        print("Warning: DataFrame is empty")
        return True
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Error: Missing required columns: {missing_columns}")
            return False
    
    return True

def clean_numeric_columns(df, columns=None):
    """
    Clean numeric columns by converting to appropriate types.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        columns (list, optional): Specific columns to clean
    
    Returns:
        pd.DataFrame: DataFrame with cleaned numeric columns
    """
    if columns is None:
        columns = df.select_dtypes(include=['object']).columns
    
    cleaned_df = df.copy()
    
    for col in columns:
        if col in df.columns:
            cleaned_df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return cleaned_df

def get_data_summary(df):
    """
    Generate summary statistics for DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame
    
    Returns:
        dict: Dictionary containing summary statistics
    """
    summary = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': df.isnull().sum().sum(),
        'duplicate_rows': df.duplicated().sum(),
        'column_types': df.dtypes.to_dict(),
        'numeric_columns': df.select_dtypes(include=['number']).columns.tolist(),
        'categorical_columns': df.select_dtypes(include=['object']).columns.tolist()
    }
    
    return summary
import numpy as np

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a pandas DataFrame column using the IQR method.
    
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

def calculate_summary_stats(data, column):
    """
    Calculate summary statistics for a column after outlier removal.
    
    Parameters:
    data (pd.DataFrame): The input DataFrame.
    column (str): The column name to analyze.
    
    Returns:
    dict: Dictionary containing count, mean, std, min, and max.
    """
    stats = {
        'count': data[column].count(),
        'mean': data[column].mean(),
        'std': data[column].std(),
        'min': data[column].min(),
        'max': data[column].max()
    }
    return stats