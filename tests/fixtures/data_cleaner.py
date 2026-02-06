
import pandas as pd
import numpy as np

def clean_dataset(df, drop_duplicates=True, standardize_columns=True):
    """
    Clean a pandas DataFrame by removing duplicates and standardizing column names.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean
    drop_duplicates (bool): Whether to remove duplicate rows
    standardize_columns (bool): Whether to standardize column names
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        removed = initial_rows - len(cleaned_df)
        print(f"Removed {removed} duplicate rows")
    
    if standardize_columns:
        cleaned_df.columns = cleaned_df.columns.str.strip().str.lower().str.replace(' ', '_')
        print("Standardized column names")
    
    return cleaned_df

def validate_data(df, required_columns=None, check_nulls=True):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate
    required_columns (list): List of required column names
    check_nulls (bool): Whether to check for null values
    
    Returns:
    dict: Validation results
    """
    validation_results = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_columns': [],
        'null_counts': {}
    }
    
    if required_columns:
        missing = [col for col in required_columns if col not in df.columns]
        validation_results['missing_columns'] = missing
    
    if check_nulls:
        null_counts = df.isnull().sum()
        validation_results['null_counts'] = null_counts[null_counts > 0].to_dict()
    
    return validation_results

def remove_outliers_iqr(df, columns, multiplier=1.5):
    """
    Remove outliers using the Interquartile Range method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    columns (list): List of column names to check for outliers
    multiplier (float): IQR multiplier for outlier detection
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed
    """
    df_clean = df.copy()
    
    for column in columns:
        if column in df_clean.columns:
            Q1 = df_clean[column].quantile(0.25)
            Q3 = df_clean[column].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - multiplier * IQR
            upper_bound = Q3 + multiplier * IQR
            
            mask = (df_clean[column] >= lower_bound) & (df_clean[column] <= upper_bound)
            df_clean = df_clean[mask]
    
    return df_clean

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'Name': ['Alice', 'Bob', 'Alice', 'Charlie', 'David'],
        'Age': [25, 30, 25, 35, 40],
        'Score': [85.5, 92.0, 85.5, 78.5, 95.0]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    
    cleaned = clean_dataset(df)
    print("\nCleaned DataFrame:")
    print(cleaned)
    
    validation = validate_data(cleaned, required_columns=['name', 'age', 'score'])
    print("\nValidation Results:")
    print(validation)