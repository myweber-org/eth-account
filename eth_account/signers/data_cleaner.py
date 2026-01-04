import pandas as pd
import numpy as np

def clean_csv_data(filepath, fill_strategy='mean', columns_to_drop=None):
    """
    Load and clean a CSV file by handling missing values and optionally dropping columns.
    
    Parameters:
    filepath (str): Path to the CSV file.
    fill_strategy (str): Strategy for filling missing values. Options: 'mean', 'median', 'mode', 'zero'.
    columns_to_drop (list): List of column names to drop from the dataset.
    
    Returns:
    pandas.DataFrame: Cleaned DataFrame.
    """
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found at path: {filepath}")
    
    original_shape = df.shape
    print(f"Original data shape: {original_shape}")
    
    if columns_to_drop:
        df = df.drop(columns=columns_to_drop, errors='ignore')
        print(f"Dropped columns: {columns_to_drop}")
    
    missing_before = df.isnull().sum().sum()
    print(f"Missing values before cleaning: {missing_before}")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if fill_strategy == 'mean':
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    elif fill_strategy == 'median':
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    elif fill_strategy == 'mode':
        for col in numeric_cols:
            df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 0)
    elif fill_strategy == 'zero':
        df[numeric_cols] = df[numeric_cols].fillna(0)
    else:
        raise ValueError(f"Unsupported fill strategy: {fill_strategy}")
    
    missing_after = df.isnull().sum().sum()
    print(f"Missing values after cleaning: {missing_after}")
    
    final_shape = df.shape
    print(f"Final data shape: {final_shape}")
    
    return df

def save_cleaned_data(df, output_path):
    """
    Save the cleaned DataFrame to a CSV file.
    
    Parameters:
    df (pandas.DataFrame): Cleaned DataFrame.
    output_path (str): Path to save the cleaned CSV file.
    """
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to: {output_path}")

if __name__ == "__main__":
    input_file = "sample_data.csv"
    output_file = "cleaned_data.csv"
    
    try:
        cleaned_df = clean_csv_data(input_file, fill_strategy='median', columns_to_drop=['id', 'unused_column'])
        save_cleaned_data(cleaned_df, output_file)
    except Exception as e:
        print(f"Error during data cleaning: {e}")