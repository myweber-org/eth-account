import pandas as pd
import numpy as np
from scipy import stats

def load_and_clean_data(filepath):
    """
    Load a CSV file and perform basic data cleaning.
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
    duplicates_removed = initial_rows - df.shape[0]
    print(f"Removed {duplicates_removed} duplicate rows.")

    # Handle missing values: drop rows where all values are NaN
    df.dropna(how='all', inplace=True)
    # For numeric columns, fill missing values with column median
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].median(), inplace=True)

    # Remove outliers using Z-score for numeric columns
    # Consider values with |Z| > 3 as outliers
    z_scores = np.abs(stats.zscore(df[numeric_cols]))
    outlier_mask = (z_scores < 3).all(axis=1)
    df_clean = df[outlier_mask].copy()
    outliers_removed = df.shape[0] - df_clean.shape[0]
    print(f"Removed {outliers_removed} outliers based on Z-score.")

    # Normalize numeric columns to range [0, 1]
    for col in numeric_cols:
        if df_clean[col].nunique() > 1:  # Avoid normalizing constant columns
            min_val = df_clean[col].min()
            max_val = df_clean[col].max()
            if max_val != min_val:
                df_clean[col] = (df_clean[col] - min_val) / (max_val - min_val)
            else:
                df_clean[col] = 0.5  # Assign middle value if constant

    print(f"Final cleaned data shape: {df_clean.shape}")
    return df_clean

def save_cleaned_data(df, output_path):
    """
    Save the cleaned DataFrame to a CSV file.
    """
    if df is not None:
        try:
            df.to_csv(output_path, index=False)
            print(f"Cleaned data saved to {output_path}")
        except Exception as e:
            print(f"Error saving file: {e}")
    else:
        print("No data to save.")

if __name__ == "__main__":
    # Example usage
    input_file = "raw_data.csv"
    output_file = "cleaned_data.csv"

    cleaned_df = load_and_clean_data(input_file)
    save_cleaned_data(cleaned_df, output_file)
def remove_duplicates_preserve_order(sequence):
    seen = set()
    result = []
    for item in sequence:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result