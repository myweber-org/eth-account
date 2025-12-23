
import pandas as pd
import numpy as np

def clean_dataframe(df, drop_duplicates=True, fill_missing=True, fill_value=0):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    if fill_missing:
        cleaned_df = cleaned_df.fillna(fill_value)
    
    return cleaned_df

def remove_outliers_iqr(df, column, multiplier=1.5):
    """
    Remove outliers from a specific column using the IQR method.
    """
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr
    
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return filtered_df

def normalize_column(df, column):
    """
    Normalize a column using min-max scaling.
    """
    min_val = df[column].min()
    max_val = df[column].max()
    
    if max_val != min_val:
        df[column + '_normalized'] = (df[column] - min_val) / (max_val - min_val)
    else:
        df[column + '_normalized'] = 0
    
    return df

def main():
    # Example usage
    data = {
        'id': [1, 2, 3, 4, 5, 5, 6],
        'value': [10, 20, np.nan, 40, 50, 50, 200],
        'category': ['A', 'B', 'A', 'B', 'A', 'A', 'C']
    }
    
    df = pd.DataFrame(data)
    print("Original DataFrame:")
    print(df)
    
    # Clean the data
    cleaned_df = clean_dataframe(df)
    print("\nCleaned DataFrame:")
    print(cleaned_df)
    
    # Remove outliers
    outlier_removed = remove_outliers_iqr(cleaned_df, 'value')
    print("\nDataFrame after outlier removal:")
    print(outlier_removed)
    
    # Normalize column
    normalized_df = normalize_column(cleaned_df.copy(), 'value')
    print("\nDataFrame with normalized column:")
    print(normalized_df)

if __name__ == "__main__":
    main()
import pandas as pd
import numpy as np
from scipy import stats

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
    df[column + '_normalized'] = (df[column] - min_val) / (max_val - min_val)
    return df

def clean_dataset(file_path):
    df = pd.read_csv(file_path)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        df = remove_outliers_iqr(df, col)
        df = normalize_minmax(df, col)
    
    return df

if __name__ == "__main__":
    cleaned_data = clean_dataset('raw_data.csv')
    cleaned_data.to_csv('cleaned_data.csv', index=False)
    print(f"Data cleaning complete. Original shape: {pd.read_csv('raw_data.csv').shape}, Cleaned shape: {cleaned_data.shape}")