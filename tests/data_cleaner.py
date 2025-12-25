
import pandas as pd
import numpy as np
from pathlib import Path

def clean_dataset(input_path, output_path=None):
    """
    Load a CSV dataset, remove duplicate rows, standardize string columns,
    handle missing values, and save the cleaned version.
    """
    df = pd.read_csv(input_path)
    
    initial_rows = df.shape[0]
    print(f"Initial dataset rows: {initial_rows}")
    
    df = df.drop_duplicates()
    duplicates_removed = initial_rows - df.shape[0]
    print(f"Removed {duplicates_removed} duplicate rows")
    
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].str.strip().str.lower()
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    
    if output_path is None:
        input_stem = Path(input_path).stem
        output_path = f"{input_stem}_cleaned.csv"
    
    df.to_csv(output_path, index=False)
    print(f"Cleaned dataset saved to: {output_path}")
    print(f"Final dataset rows: {df.shape[0]}, columns: {df.shape[1]}")
    
    return df

if __name__ == "__main__":
    sample_data = {
        'name': ['Alice', 'alice ', 'Bob', 'bob', 'Charlie', None],
        'age': [25, 25, 30, 30, 35, 40],
        'score': [85.5, 85.5, 92.0, None, 78.5, 88.0]
    }
    
    test_df = pd.DataFrame(sample_data)
    test_df.to_csv('test_data.csv', index=False)
    
    cleaned_df = clean_dataset('test_data.csv')
    print("\nSample cleaned data:")
    print(cleaned_df.head())