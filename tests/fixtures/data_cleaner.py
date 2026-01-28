import pandas as pd
import numpy as np
from scipy import stats

def load_and_clean_data(filepath):
    """
    Load a CSV file and perform basic data cleaning.
    """
    try:
        df = pd.read_csv(filepath)
        print(f"Loaded data with shape: {df.shape}")
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

    # Handle missing values: fill numeric columns with median, drop rows for categorical if too many missing
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            print(f"Filled missing values in {col} with median: {median_val}")

    categorical_cols = df.select_dtypes(exclude=[np.number]).columns
    for col in categorical_cols:
        missing_count = df[col].isnull().sum()
        if missing_count > 0:
            if missing_count / len(df) < 0.1:  # Less than 10% missing
                mode_val = df[col].mode()[0]
                df[col].fillna(mode_val, inplace=True)
                print(f"Filled missing values in {col} with mode: {mode_val}")
            else:
                df.dropna(subset=[col], inplace=True)
                print(f"Dropped rows with missing values in {col}")

    # Remove outliers using Z-score for numeric columns (optional, based on threshold)
    z_threshold = 3
    for col in numeric_cols:
        z_scores = np.abs(stats.zscore(df[col]))
        outlier_indices = np.where(z_scores > z_threshold)[0]
        if len(outlier_indices) > 0:
            df = df.drop(df.index[outlier_indices])
            print(f"Removed {len(outlier_indices)} outliers from {col} based on Z-score > {z_threshold}")

    # Normalize numeric columns to range [0, 1] (optional)
    for col in numeric_cols:
        if df[col].max() - df[col].min() > 0:  # Avoid division by zero
            df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
            print(f"Normalized column {col} to range [0, 1]")

    print(f"Final data shape: {df.shape}")
    return df

def save_cleaned_data(df, output_filepath):
    """
    Save the cleaned DataFrame to a CSV file.
    """
    if df is not None:
        df.to_csv(output_filepath, index=False)
        print(f"Cleaned data saved to {output_filepath}")
    else:
        print("No data to save.")

if __name__ == "__main__":
    input_file = "raw_data.csv"
    output_file = "cleaned_data.csv"

    cleaned_df = load_and_clean_data(input_file)
    save_cleaned_data(cleaned_df, output_file)