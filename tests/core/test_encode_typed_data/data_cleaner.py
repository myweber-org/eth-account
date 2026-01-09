import pandas as pd
import numpy as np

def clean_missing_data(filepath, strategy='mean', columns=None):
    """
    Load a CSV file and handle missing values using specified strategy.
    
    Parameters:
    filepath (str): Path to the CSV file.
    strategy (str): Method for handling missing values ('mean', 'median', 'mode', 'drop').
    columns (list): Specific columns to apply cleaning. If None, applies to all numeric columns.
    
    Returns:
    pandas.DataFrame: Cleaned DataFrame.
    """
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found at {filepath}")
    
    if columns is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        columns = list(numeric_cols)
    
    for col in columns:
        if col not in df.columns:
            continue
        
        if strategy == 'mean':
            fill_value = df[col].mean()
        elif strategy == 'median':
            fill_value = df[col].median()
        elif strategy == 'mode':
            fill_value = df[col].mode()[0] if not df[col].mode().empty else np.nan
        elif strategy == 'drop':
            df = df.dropna(subset=[col])
            continue
        else:
            raise ValueError(f"Unsupported strategy: {strategy}")
        
        df[col] = df[col].fillna(fill_value)
    
    return df

def save_cleaned_data(df, output_path):
    """
    Save cleaned DataFrame to CSV.
    
    Parameters:
    df (pandas.DataFrame): DataFrame to save.
    output_path (str): Path for output CSV file.
    """
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to {output_path}")

if __name__ == "__main__":
    input_file = "raw_data.csv"
    output_file = "cleaned_data.csv"
    
    cleaned_df = clean_missing_data(input_file, strategy='median')
    save_cleaned_data(cleaned_df, output_file)