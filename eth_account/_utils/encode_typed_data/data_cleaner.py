import numpy as np
import pandas as pd

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def normalize_minmax(df, column):
    min_val = df[column].min()
    max_val = df[column].max()
    if max_val == min_val:
        return df[column].apply(lambda x: 0.0)
    return df[column].apply(lambda x: (x - min_val) / (max_val - min_val))

def clean_dataset(df, numeric_columns):
    cleaned_df = df.copy()
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
            cleaned_df[col] = normalize_minmax(cleaned_df, col)
    return cleaned_df.reset_index(drop=True)

def validate_data(df, required_columns):
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    return Trueimport pandas as pd

def clean_dataset(df, drop_duplicates=True, fill_method=None):
    """
    Clean a pandas DataFrame by handling missing values and optionally removing duplicates.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    drop_duplicates (bool): If True, remove duplicate rows.
    fill_method (str or None): Method to fill missing values. 
                               Options: 'ffill', 'bfill', 'mean', 'median', or None to drop rows.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    # Handle missing values
    if fill_method is None:
        cleaned_df = cleaned_df.dropna()
    elif fill_method in ['ffill', 'bfill']:
        cleaned_df = cleaned_df.fillna(method=fill_method)
    elif fill_method == 'mean':
        cleaned_df = cleaned_df.fillna(cleaned_df.mean(numeric_only=True))
    elif fill_method == 'median':
        cleaned_df = cleaned_df.fillna(cleaned_df.median(numeric_only=True))
    
    # Remove duplicates if requested
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    # Reset index after cleaning
    cleaned_df = cleaned_df.reset_index(drop=True)
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of column names that must be present.
    
    Returns:
    tuple: (is_valid, error_message)
    """
    if not isinstance(df, pd.DataFrame):
        return False, "Input is not a pandas DataFrame"
    
    if df.empty:
        return False, "DataFrame is empty"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    return True, "DataFrame is valid"

# Example usage (commented out for production)
# if __name__ == "__main__":
#     sample_data = {
#         'A': [1, 2, None, 4, 4],
#         'B': [5, None, 7, 8, 8],
#         'C': [9, 10, 11, None, 11]
#     }
#     df = pd.DataFrame(sample_data)
#     print("Original DataFrame:")
#     print(df)
#     
#     cleaned = clean_dataset(df, fill_method='mean')
#     print("\nCleaned DataFrame:")
#     print(cleaned)
#     
#     is_valid, message = validate_dataframe(cleaned, required_columns=['A', 'B', 'C'])
#     print(f"\nValidation: {is_valid}, Message: {message}")