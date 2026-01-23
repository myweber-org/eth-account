import pandas as pd
import numpy as np

def clean_csv_data(filepath, fill_strategy='mean', drop_threshold=0.5):
    """
    Load and clean CSV data by handling missing values and removing invalid columns.
    
    Parameters:
    filepath (str): Path to the CSV file.
    fill_strategy (str): Strategy for filling missing values ('mean', 'median', 'mode', or 'zero').
    drop_threshold (float): Drop columns with missing ratio above this threshold (0.0 to 1.0).
    
    Returns:
    pandas.DataFrame: Cleaned DataFrame.
    """
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {filepath}")
    
    print(f"Original shape: {df.shape}")
    
    # Drop columns with too many missing values
    missing_ratio = df.isnull().sum() / len(df)
    columns_to_drop = missing_ratio[missing_ratio > drop_threshold].index
    df = df.drop(columns=columns_to_drop)
    
    if len(columns_to_drop) > 0:
        print(f"Dropped columns: {list(columns_to_drop)}")
    
    # Fill missing values based on strategy
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if fill_strategy == 'mean':
        fill_values = df[numeric_cols].mean()
    elif fill_strategy == 'median':
        fill_values = df[numeric_cols].median()
    elif fill_strategy == 'mode':
        fill_values = df[numeric_cols].mode().iloc[0]
    elif fill_strategy == 'zero':
        fill_values = 0
    else:
        raise ValueError("fill_strategy must be 'mean', 'median', 'mode', or 'zero'")
    
    df[numeric_cols] = df[numeric_cols].fillna(fill_values)
    
    # For non-numeric columns, fill with most frequent value
    non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns
    for col in non_numeric_cols:
        most_frequent = df[col].mode()[0] if not df[col].mode().empty else ''
        df[col] = df[col].fillna(most_frequent)
    
    print(f"Cleaned shape: {df.shape}")
    print(f"Remaining missing values: {df.isnull().sum().sum()}")
    
    return df

def export_cleaned_data(df, output_path):
    """
    Export cleaned DataFrame to CSV.
    
    Parameters:
    df (pandas.DataFrame): Cleaned DataFrame.
    output_path (str): Path for output CSV file.
    """
    df.to_csv(output_path, index=False)
    print(f"Cleaned data exported to: {output_path}")

if __name__ == "__main__":
    # Example usage
    input_file = "raw_data.csv"
    output_file = "cleaned_data.csv"
    
    try:
        cleaned_df = clean_csv_data(input_file, fill_strategy='median', drop_threshold=0.3)
        export_cleaned_data(cleaned_df, output_file)
    except Exception as e:
        print(f"Error during data cleaning: {e}")