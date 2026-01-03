import pandas as pd
import numpy as np

def load_and_clean_data(filepath):
    """
    Load a CSV file and perform basic data cleaning operations.
    """
    try:
        df = pd.read_csv(filepath)
        print(f"Data loaded successfully. Shape: {df.shape}")
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None
    except Exception as e:
        print(f"Error loading file: {e}")
        return None

    # Remove duplicate rows
    initial_rows = df.shape[0]
    df.drop_duplicates(inplace=True)
    removed_duplicates = initial_rows - df.shape[0]
    if removed_duplicates > 0:
        print(f"Removed {removed_duplicates} duplicate rows.")

    # Handle missing values: drop rows where all values are NaN
    df.dropna(how='all', inplace=True)
    # For numeric columns, fill missing values with column median
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].apply(lambda col: col.fillna(col.median()))

    # Remove outliers using IQR method for numeric columns
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        # Filter out outliers
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

    print(f"Data after cleaning. New shape: {df.shape}")
    return df

def normalize_numeric_columns(df):
    """
    Normalize numeric columns to range [0, 1] using min-max scaling.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        min_val = df[col].min()
        max_val = df[col].max()
        if max_val > min_val:  # Avoid division by zero
            df[col] = (df[col] - min_val) / (max_val - min_val)
        else:
            df[col] = 0  # If all values are the same, set to 0
    print("Numeric columns normalized.")
    return df

if __name__ == "__main__":
    # Example usage
    cleaned_df = load_and_clean_data("sample_data.csv")
    if cleaned_df is not None:
        normalized_df = normalize_numeric_columns(cleaned_df)
        print("Data cleaning and normalization complete.")
        print(normalized_df.head())