import pandas as pd
import numpy as np

def load_data(filepath):
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(filepath)
        print(f"Data loaded successfully. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None

def remove_outliers(df, column, threshold=3):
    """Remove outliers using the Z-score method."""
    if column not in df.columns:
        print(f"Column '{column}' not found in DataFrame.")
        return df
    
    z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
    df_clean = df[z_scores < threshold]
    removed_count = len(df) - len(df_clean)
    print(f"Removed {removed_count} outliers from column '{column}'.")
    return df_clean

def normalize_column(df, column):
    """Normalize a column to range [0, 1]."""
    if column not in df.columns:
        print(f"Column '{column}' not found in DataFrame.")
        return df
    
    min_val = df[column].min()
    max_val = df[column].max()
    
    if max_val == min_val:
        print(f"Column '{column}' has constant values. Normalization skipped.")
        return df
    
    df[column] = (df[column] - min_val) / (max_val - min_val)
    print(f"Column '{column}' normalized to range [0, 1].")
    return df

def clean_data(df, numeric_columns):
    """Main function to clean the dataset."""
    if df is None or df.empty:
        print("DataFrame is empty or None.")
        return df
    
    df_clean = df.copy()
    
    for col in numeric_columns:
        if col in df_clean.columns:
            df_clean = remove_outliers(df_clean, col)
            df_clean = normalize_column(df_clean, col)
        else:
            print(f"Skipping column '{col}' as it does not exist.")
    
    print(f"Data cleaning complete. Final shape: {df_clean.shape}")
    return df_clean

if __name__ == "__main__":
    data_path = "sample_data.csv"
    numeric_cols = ["feature1", "feature2", "feature3"]
    
    raw_data = load_data(data_path)
    if raw_data is not None:
        cleaned_data = clean_data(raw_data, numeric_cols)
        cleaned_data.to_csv("cleaned_data.csv", index=False)
        print("Cleaned data saved to 'cleaned_data.csv'.")