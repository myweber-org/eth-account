def remove_duplicates(sequence):
    seen = set()
    result = []
    for item in sequence:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return resultimport pandas as pd
import numpy as np

def clean_csv_data(filepath, fill_method='mean', drop_threshold=0.5):
    """
    Load and clean a CSV file by handling missing values.
    
    Parameters:
    filepath (str): Path to the CSV file.
    fill_method (str): Method for filling missing values ('mean', 'median', 'mode', or 'zero').
    drop_threshold (float): Drop columns with missing ratio above this threshold (0.0 to 1.0).
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {filepath}")
    
    original_shape = df.shape
    print(f"Original data shape: {original_shape}")
    
    missing_ratio = df.isnull().sum() / len(df)
    columns_to_drop = missing_ratio[missing_ratio > drop_threshold].index
    df = df.drop(columns=columns_to_drop)
    
    if len(columns_to_drop) > 0:
        print(f"Dropped columns with >{drop_threshold*100}% missing values: {list(columns_to_drop)}")
    
    for column in df.columns:
        if df[column].dtype in ['int64', 'float64']:
            if fill_method == 'mean':
                fill_value = df[column].mean()
            elif fill_method == 'median':
                fill_value = df[column].median()
            elif fill_method == 'mode':
                fill_value = df[column].mode()[0] if not df[column].mode().empty else 0
            elif fill_method == 'zero':
                fill_value = 0
            else:
                raise ValueError("fill_method must be 'mean', 'median', 'mode', or 'zero'")
            
            df[column].fillna(fill_value, inplace=True)
        else:
            df[column].fillna(df[column].mode()[0] if not df[column].mode().empty else 'Unknown', inplace=True)
    
    cleaned_shape = df.shape
    print(f"Cleaned data shape: {cleaned_shape}")
    print(f"Removed {original_shape[1] - cleaned_shape[1]} columns, {original_shape[0] - cleaned_shape[0]} rows")
    
    return df

def detect_outliers_iqr(df, column):
    """
    Detect outliers in a numeric column using IQR method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame.
    column (str): Column name to check for outliers.
    
    Returns:
    pd.Series: Boolean series indicating outliers.
    """
    if df[column].dtype not in ['int64', 'float64']:
        raise TypeError(f"Column '{column}' must be numeric")
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = (df[column] < lower_bound) | (df[column] > upper_bound)
    return outliers

def save_cleaned_data(df, output_path):
    """
    Save cleaned DataFrame to CSV.
    
    Parameters:
    df (pd.DataFrame): Cleaned DataFrame.
    output_path (str): Path to save the cleaned CSV.
    """
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to: {output_path}")

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'A': [1, 2, np.nan, 4, 5],
        'B': [np.nan, np.nan, np.nan, 10, 11],
        'C': [5, 6, 7, 8, 9],
        'D': ['X', 'Y', np.nan, 'Z', 'X']
    })
    
    sample_data.to_csv('sample_data.csv', index=False)
    
    cleaned_df = clean_csv_data('sample_data.csv', fill_method='median', drop_threshold=0.6)
    print("\nSample cleaned data:")
    print(cleaned_df)
    
    outliers = detect_outliers_iqr(cleaned_df, 'C')
    print(f"\nOutliers in column 'C': {outliers.sum()}")
    
    save_cleaned_data(cleaned_df, 'cleaned_sample_data.csv')
import pandas as pd
import numpy as np
from scipy import stats

def load_and_clean_data(filepath):
    df = pd.read_csv(filepath)
    
    # Remove duplicate rows
    df = df.drop_duplicates()
    
    # Handle missing values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    
    # Remove outliers using z-score
    z_scores = np.abs(stats.zscore(df[numeric_cols]))
    df = df[(z_scores < 3).all(axis=1)]
    
    # Normalize numeric columns
    df[numeric_cols] = (df[numeric_cols] - df[numeric_cols].mean()) / df[numeric_cols].std()
    
    return df

def save_cleaned_data(df, output_path):
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to {output_path}")

if __name__ == "__main__":
    input_file = "raw_data.csv"
    output_file = "cleaned_data.csv"
    
    cleaned_df = load_and_clean_data(input_file)
    save_cleaned_data(cleaned_df, output_file)