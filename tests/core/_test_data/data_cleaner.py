
import pandas as pd
import numpy as np
from scipy import stats

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def normalize_minmax(df, column):
    min_val = df[column].min()
    max_val = df[column].max()
    df[column + '_normalized'] = (df[column] - min_val) / (max_val - min_val)
    return df

def clean_dataset(file_path, columns_to_clean):
    df = pd.read_csv(file_path)
    
    for column in columns_to_clean:
        if column in df.columns:
            df = remove_outliers_iqr(df, column)
            df = normalize_minmax(df, column)
    
    cleaned_file = file_path.replace('.csv', '_cleaned.csv')
    df.to_csv(cleaned_file, index=False)
    return cleaned_file

if __name__ == "__main__":
    input_file = "raw_data.csv"
    columns = ["temperature", "humidity", "pressure"]
    result = clean_dataset(input_file, columns)
    print(f"Cleaned data saved to: {result}")
def remove_duplicates(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]import pandas as pd
import numpy as np
from scipy import stats

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
    initial_count = len(df)
    df.drop_duplicates(inplace=True)
    removed_duplicates = initial_count - len(df)
    print(f"Removed {removed_duplicates} duplicate rows.")

    # Handle missing values: drop rows where all values are NaN
    df.dropna(how='all', inplace=True)
    # For numeric columns, fill missing values with the column median
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].apply(lambda col: col.fillna(col.median()))

    # Remove outliers using Z-score for numeric columns
    # Consider points with |Z| > 3 as outliers
    z_scores = np.abs(stats.zscore(df[numeric_cols], nan_policy='omit'))
    outlier_mask = (z_scores < 3).all(axis=1)
    df_clean = df[outlier_mask].copy()
    removed_outliers = len(df) - len(df_clean)
    print(f"Removed {removed_outliers} rows based on Z-score outlier detection (|Z| > 3).")

    # Normalize numeric columns to range [0, 1]
    if not df_clean[numeric_cols].empty:
        df_clean[numeric_cols] = (df_clean[numeric_cols] - df_clean[numeric_cols].min()) / (df_clean[numeric_cols].max() - df_clean[numeric_cols].min())
        print("Normalized numeric columns to range [0, 1].")

    print(f"Final cleaned data shape: {df_clean.shape}")
    return df_clean

def save_cleaned_data(df, output_filepath):
    """
    Save the cleaned DataFrame to a CSV file.
    """
    if df is not None:
        try:
            df.to_csv(output_filepath, index=False)
            print(f"Cleaned data saved to {output_filepath}")
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