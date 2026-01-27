import pandas as pd
import numpy as np

def clean_csv_data(filepath, fill_method='mean', remove_duplicates=True):
    """
    Load and clean CSV data by handling missing values and duplicates.
    
    Args:
        filepath (str): Path to the CSV file.
        fill_method (str): Method for filling missing values ('mean', 'median', 'mode').
        remove_duplicates (bool): Whether to remove duplicate rows.
    
    Returns:
        pandas.DataFrame: Cleaned DataFrame.
    """
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {filepath}")
    
    original_shape = df.shape
    
    if remove_duplicates:
        df = df.drop_duplicates()
        print(f"Removed {original_shape[0] - df.shape[0]} duplicate rows.")
    
    if df.isnull().sum().any():
        print("Handling missing values...")
        for column in df.select_dtypes(include=[np.number]).columns:
            if df[column].isnull().any():
                if fill_method == 'mean':
                    fill_value = df[column].mean()
                elif fill_method == 'median':
                    fill_value = df[column].median()
                elif fill_method == 'mode':
                    fill_value = df[column].mode()[0]
                else:
                    fill_value = 0
                
                df[column].fillna(fill_value, inplace=True)
                print(f"Filled missing values in '{column}' with {fill_method}: {fill_value}")
    
    print(f"Data cleaning complete. Original shape: {original_shape}, Cleaned shape: {df.shape}")
    return df

def save_cleaned_data(df, output_path):
    """
    Save cleaned DataFrame to a new CSV file.
    
    Args:
        df (pandas.DataFrame): Cleaned DataFrame.
        output_path (str): Path for the output CSV file.
    """
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to: {output_path}")

if __name__ == "__main__":
    input_file = "raw_data.csv"
    output_file = "cleaned_data.csv"
    
    try:
        cleaned_df = clean_csv_data(input_file, fill_method='median')
        save_cleaned_data(cleaned_df, output_file)
    except Exception as e:
        print(f"Error during data cleaning: {e}")