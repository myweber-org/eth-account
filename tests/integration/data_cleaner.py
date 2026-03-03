import pandas as pd
import numpy as np

def load_data(filepath):
    """Load dataset from CSV file."""
    try:
        df = pd.read_csv(filepath)
        print(f"Data loaded successfully. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found.")
        return None

def remove_outliers_iqr(df, column):
    """Remove outliers using IQR method for specified column."""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    removed_count = len(df) - len(filtered_df)
    print(f"Removed {removed_count} outliers from column '{column}'")
    
    return filtered_df

def normalize_column(df, column):
    """Normalize column values to range [0, 1]."""
    if column not in df.columns:
        print(f"Error: Column '{column}' not found in dataframe.")
        return df
    
    min_val = df[column].min()
    max_val = df[column].max()
    
    if max_val == min_val:
        print(f"Warning: Column '{column}' has constant values. Normalization skipped.")
        return df
    
    df[column + '_normalized'] = (df[column] - min_val) / (max_val - min_val)
    print(f"Column '{column}' normalized successfully.")
    
    return df

def clean_dataset(df, numeric_columns):
    """Main cleaning pipeline for numeric columns."""
    if df is None or df.empty:
        print("Error: Invalid dataframe provided.")
        return df
    
    original_shape = df.shape
    
    for column in numeric_columns:
        if column in df.columns:
            df = remove_outliers_iqr(df, column)
            df = normalize_column(df, column)
        else:
            print(f"Warning: Column '{column}' not found. Skipping.")
    
    print(f"Data cleaning complete. Original shape: {original_shape}, Cleaned shape: {df.shape}")
    return df

def save_cleaned_data(df, output_path):
    """Save cleaned dataframe to CSV."""
    if df is not None and not df.empty:
        df.to_csv(output_path, index=False)
        print(f"Cleaned data saved to: {output_path}")
        return True
    else:
        print("Error: No data to save.")
        return False

if __name__ == "__main__":
    input_file = "raw_data.csv"
    output_file = "cleaned_data.csv"
    
    numeric_cols = ['age', 'income', 'score']
    
    raw_data = load_data(input_file)
    
    if raw_data is not None:
        cleaned_data = clean_dataset(raw_data, numeric_cols)
        save_cleaned_data(cleaned_data, output_file)