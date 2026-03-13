
import pandas as pd
import numpy as np

def clean_dataset(df, drop_duplicates=True, fill_missing=True, fill_strategy='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    drop_duplicates (bool): Whether to drop duplicate rows.
    fill_missing (bool): Whether to fill missing values.
    fill_strategy (str): Strategy for filling missing values ('mean', 'median', 'mode', or 'constant').
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        removed = initial_rows - len(cleaned_df)
        print(f"Removed {removed} duplicate rows.")
    
    if fill_missing:
        numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
        categorical_cols = cleaned_df.select_dtypes(exclude=[np.number]).columns
        
        if fill_strategy == 'mean':
            for col in numeric_cols:
                if cleaned_df[col].isnull().any():
                    cleaned_df[col].fillna(cleaned_df[col].mean(), inplace=True)
        elif fill_strategy == 'median':
            for col in numeric_cols:
                if cleaned_df[col].isnull().any():
                    cleaned_df[col].fillna(cleaned_df[col].median(), inplace=True)
        elif fill_strategy == 'mode':
            for col in cleaned_df.columns:
                if cleaned_df[col].isnull().any():
                    cleaned_df[col].fillna(cleaned_df[col].mode()[0], inplace=True)
        elif fill_strategy == 'constant':
            for col in cleaned_df.columns:
                if cleaned_df[col].isnull().any():
                    if col in numeric_cols:
                        cleaned_df[col].fillna(0, inplace=True)
                    else:
                        cleaned_df[col].fillna('unknown', inplace=True)
        
        missing_after = cleaned_df.isnull().sum().sum()
        if missing_after > 0:
            print(f"Warning: {missing_after} missing values remain after cleaning.")
        else:
            print("All missing values have been handled.")
    
    return cleaned_df

def validate_data(df, required_columns=None, min_rows=1):
    """
    Validate the structure and content of a DataFrame.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of column names that must be present.
    min_rows (int): Minimum number of rows required.
    
    Returns:
    bool: True if validation passes, False otherwise.
    """
    if len(df) < min_rows:
        print(f"Validation failed: DataFrame has only {len(df)} rows, minimum required is {min_rows}.")
        return False
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            print(f"Validation failed: Missing required columns: {missing_cols}")
            return False
    
    print("Data validation passed.")
    return True

def remove_outliers(df, column, method='iqr', threshold=1.5):
    """
    Remove outliers from a specific column in a DataFrame.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame.
    column (str): Column name to process.
    method (str): Method for outlier detection ('iqr' or 'zscore').
    threshold (float): Threshold for outlier detection.
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed.
    """
    if column not in df.columns:
        print(f"Column '{column}' not found in DataFrame.")
        return df
    
    if method == 'iqr':
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    elif method == 'zscore':
        from scipy import stats
        z_scores = np.abs(stats.zscore(df[column].dropna()))
        filtered_df = df[(z_scores < threshold) | (df[column].isna())]
    else:
        print(f"Unknown method: {method}. Returning original DataFrame.")
        return df
    
    removed_count = len(df) - len(filtered_df)
    print(f"Removed {removed_count} outliers from column '{column}'.")
    
    return filtered_df

if __name__ == "__main__":
    sample_data = {
        'id': [1, 2, 2, 3, 4, 5, 5],
        'value': [10, 20, 20, None, 40, 50, 1000],
        'category': ['A', 'B', 'B', 'C', None, 'A', 'A']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    cleaned = clean_dataset(df, fill_strategy='mode')
    print("\nCleaned DataFrame:")
    print(cleaned)
    
    is_valid = validate_data(cleaned, required_columns=['id', 'value'], min_rows=2)
    
    no_outliers = remove_outliers(cleaned, 'value', method='iqr')
    print("\nDataFrame after outlier removal:")
    print(no_outliers)