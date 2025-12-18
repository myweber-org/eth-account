
import pandas as pd
import numpy as np

def clean_dataset(df, columns_to_check=None, fill_strategy='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    columns_to_check (list, optional): List of column names to check for duplicates.
                                       If None, uses all columns.
    fill_strategy (str): Strategy to fill missing values. 
                         Options: 'mean', 'median', 'mode', 'drop', or a numeric value.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    # Remove duplicates
    if columns_to_check is None:
        columns_to_check = cleaned_df.columns.tolist()
    cleaned_df = cleaned_df.drop_duplicates(subset=columns_to_check, keep='first')
    
    # Handle missing values
    numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = cleaned_df.select_dtypes(exclude=[np.number]).columns.tolist()
    
    if fill_strategy == 'drop':
        cleaned_df = cleaned_df.dropna()
    elif fill_strategy in ['mean', 'median']:
        for col in numeric_cols:
            if cleaned_df[col].isnull().any():
                if fill_strategy == 'mean':
                    fill_value = cleaned_df[col].mean()
                else:
                    fill_value = cleaned_df[col].median()
                cleaned_df[col] = cleaned_df[col].fillna(fill_value)
    elif fill_strategy == 'mode':
        for col in cleaned_df.columns:
            if cleaned_df[col].isnull().any():
                mode_value = cleaned_df[col].mode()
                if not mode_value.empty:
                    cleaned_df[col] = cleaned_df[col].fillna(mode_value.iloc[0])
    elif isinstance(fill_strategy, (int, float)):
        cleaned_df = cleaned_df.fillna(fill_strategy)
    
    return cleaned_df

def validate_dataframe(df, required_columns=None, min_rows=1):
    """
    Validate basic DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of column names that must be present.
    min_rows (int): Minimum number of rows required.
    
    Returns:
    tuple: (is_valid, error_message)
    """
    if not isinstance(df, pd.DataFrame):
        return False, "Input is not a pandas DataFrame"
    
    if df.empty:
        return False, "DataFrame is empty"
    
    if len(df) < min_rows:
        return False, f"DataFrame has fewer than {min_rows} rows"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    return True, "DataFrame is valid"

def remove_outliers_iqr(df, columns=None, multiplier=1.5):
    """
    Remove outliers using the Interquartile Range (IQR) method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame.
    columns (list): List of numeric columns to check for outliers.
                    If None, uses all numeric columns.
    multiplier (float): IQR multiplier for outlier detection.
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed.
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    filtered_df = df.copy()
    for col in columns:
        if col in filtered_df.columns and pd.api.types.is_numeric_dtype(filtered_df[col]):
            Q1 = filtered_df[col].quantile(0.25)
            Q3 = filtered_df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - multiplier * IQR
            upper_bound = Q3 + multiplier * IQR
            
            # Keep only non-outliers
            filtered_df = filtered_df[(filtered_df[col] >= lower_bound) & 
                                      (filtered_df[col] <= upper_bound)]
    
    return filtered_df