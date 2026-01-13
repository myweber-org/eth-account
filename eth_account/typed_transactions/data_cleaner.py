
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to process
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
    return filtered_df

def clean_dataset(df, numeric_columns=None):
    """
    Clean dataset by removing outliers from all numeric columns.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    numeric_columns (list): List of numeric column names. If None, uses all numeric columns.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_df = df.copy()
    
    for column in numeric_columns:
        if column in df.columns:
            original_count = len(cleaned_df)
            cleaned_df = remove_outliers_iqr(cleaned_df, column)
            removed_count = original_count - len(cleaned_df)
            print(f"Removed {removed_count} outliers from column '{column}'")
    
    return cleaned_df

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'A': np.random.normal(100, 15, 1000),
        'B': np.random.exponential(50, 1000),
        'C': np.random.uniform(0, 200, 1000)
    }
    
    sample_df = pd.DataFrame(sample_data)
    sample_df.loc[::100, 'A'] = 500  # Add some outliers
    
    print(f"Original shape: {sample_df.shape}")
    cleaned_df = clean_dataset(sample_df)
    print(f"Cleaned shape: {cleaned_df.shape}")
import pandas as pd
import numpy as np
import sys

def load_csv(filepath):
    try:
        df = pd.read_csv(filepath)
        print(f"Loaded {len(df)} rows from {filepath}")
        return df
    except FileNotFoundError:
        print(f"Error: File {filepath} not found")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading CSV: {e}")
        sys.exit(1)

def clean_data(df):
    original_rows = len(df)
    
    df.replace('', np.nan, inplace=True)
    df.replace('NA', np.nan, inplace=True)
    df.replace('null', np.nan, inplace=True)
    
    df.dropna(subset=['id'], inplace=True)
    
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        if df[col].isna().sum() > 0:
            df[col].fillna(df[col].median(), inplace=True)
    
    text_columns = df.select_dtypes(include=['object']).columns
    for col in text_columns:
        if df[col].isna().sum() > 0:
            df[col].fillna('Unknown', inplace=True)
    
    df.drop_duplicates(inplace=True)
    
    cleaned_rows = len(df)
    print(f"Cleaning complete: {original_rows} -> {cleaned_rows} rows")
    print(f"Removed {original_rows - cleaned_rows} rows")
    
    return df

def save_cleaned_data(df, output_path):
    try:
        df.to_csv(output_path, index=False)
        print(f"Cleaned data saved to {output_path}")
    except Exception as e:
        print(f"Error saving cleaned data: {e}")
        sys.exit(1)

def main():
    if len(sys.argv) != 3:
        print("Usage: python data_cleaner.py <input_file> <output_file>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    print(f"Starting data cleaning for {input_file}")
    
    df = load_csv(input_file)
    df_cleaned = clean_data(df)
    save_cleaned_data(df_cleaned, output_file)
    
    print("Data cleaning completed successfully")

if __name__ == "__main__":
    main()