
import pandas as pd

def clean_dataset(df):
    """
    Clean a pandas DataFrame by removing null values and duplicates.
    
    Args:
        df (pd.DataFrame): Input DataFrame to be cleaned.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    
    cleaned_df = df.copy()
    
    cleaned_df = cleaned_df.dropna()
    
    cleaned_df = cleaned_df.drop_duplicates()
    
    cleaned_df = cleaned_df.reset_index(drop=True)
    
    return cleaned_df

def validate_clean_dataset(df):
    """
    Validate that the DataFrame has no null values or duplicates.
    
    Args:
        df (pd.DataFrame): DataFrame to validate.
    
    Returns:
        bool: True if DataFrame is clean, False otherwise.
    """
    null_count = df.isnull().sum().sum()
    duplicate_count = df.duplicated().sum()
    
    return null_count == 0 and duplicate_count == 0

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, None, 4, 1],
        'B': [5, 6, 7, None, 5],
        'C': [8, 9, 10, 11, 8]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nNull values:", df.isnull().sum().sum())
    print("Duplicates:", df.duplicated().sum())
    
    cleaned_df = clean_dataset(df)
    print("\nCleaned DataFrame:")
    print(cleaned_df)
    print("\nValidation result:", validate_clean_dataset(cleaned_df))import pandas as pd
import numpy as np

def detect_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    return outliers

def normalize_minmax(data, column):
    min_val = data[column].min()
    max_val = data[column].max()
    if max_val - min_val == 0:
        return data[column]
    normalized = (data[column] - min_val) / (max_val - min_val)
    return normalized

def clean_dataset(df, numeric_columns):
    cleaned_df = df.copy()
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
            cleaned_df[col] = normalize_minmax(cleaned_df, col)
    return cleaned_df

def main():
    sample_data = {'values': [10, 12, 12, 13, 12, 11, 10, 100, 12, 14, 12, 10]}
    df = pd.DataFrame(sample_data)
    print("Original Data:")
    print(df)
    
    outliers = detect_outliers_iqr(df, 'values')
    print(f"\nDetected outliers:\n{outliers}")
    
    cleaned_df = clean_dataset(df, ['values'])
    print("\nCleaned Data:")
    print(cleaned_df)

if __name__ == "__main__":
    main()