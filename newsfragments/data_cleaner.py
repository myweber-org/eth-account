import pandas as pd
import numpy as np
from scipy import stats

def clean_dataset(df, numeric_columns=None, method='median', z_threshold=3):
    """
    Clean dataset by handling missing values and removing outliers.
    
    Parameters:
    df (pd.DataFrame): Input dataframe
    numeric_columns (list): List of numeric column names
    method (str): Imputation method ('median', 'mean', 'mode')
    z_threshold (float): Z-score threshold for outlier detection
    
    Returns:
    pd.DataFrame: Cleaned dataframe
    """
    if df.empty:
        return df
    
    df_clean = df.copy()
    
    if numeric_columns is None:
        numeric_columns = df_clean.select_dtypes(include=[np.number]).columns.tolist()
    
    for col in numeric_columns:
        if col not in df_clean.columns:
            continue
            
        series = df_clean[col]
        
        if series.isnull().any():
            if method == 'median':
                fill_value = series.median()
            elif method == 'mean':
                fill_value = series.mean()
            elif method == 'mode':
                fill_value = series.mode()[0] if not series.mode().empty else series.median()
            else:
                fill_value = series.median()
            
            df_clean[col] = series.fillna(fill_value)
        
        if len(series) > 10:
            z_scores = np.abs(stats.zscore(df_clean[col]))
            outlier_indices = np.where(z_scores > z_threshold)[0]
            
            if len(outlier_indices) > 0:
                median_value = df_clean[col].median()
                df_clean.loc[outlier_indices, col] = median_value
    
    return df_clean

def validate_dataframe(df, required_columns=None):
    """
    Validate dataframe structure and content.
    
    Parameters:
    df (pd.DataFrame): Dataframe to validate
    required_columns (list): List of required column names
    
    Returns:
    tuple: (is_valid, error_message)
    """
    if not isinstance(df, pd.DataFrame):
        return False, "Input is not a pandas DataFrame"
    
    if df.empty:
        return False, "DataFrame is empty"
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
    
    return True, "DataFrame is valid"

def get_data_summary(df):
    """
    Generate summary statistics for dataframe.
    
    Parameters:
    df (pd.DataFrame): Input dataframe
    
    Returns:
    dict: Summary statistics
    """
    summary = {
        'shape': df.shape,
        'columns': df.columns.tolist(),
        'dtypes': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'numeric_summary': {}
    }
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        summary['numeric_summary'][col] = {
            'mean': df[col].mean(),
            'median': df[col].median(),
            'std': df[col].std(),
            'min': df[col].min(),
            'max': df[col].max()
        }
    
    return summary

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, np.nan, 4, 100],
        'B': [5, 6, 7, np.nan, 9],
        'C': ['x', 'y', 'z', 'x', 'y']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    
    cleaned_df = clean_dataset(df, numeric_columns=['A', 'B'])
    print("\nCleaned DataFrame:")
    print(cleaned_df)
    
    is_valid, message = validate_dataframe(cleaned_df)
    print(f"\nValidation: {is_valid} - {message}")
    
    summary = get_data_summary(cleaned_df)
    print("\nData Summary:")
    print(f"Shape: {summary['shape']}")
    print(f"Numeric Summary: {summary['numeric_summary']}")