
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def normalize_column(df, column):
    min_val = df[column].min()
    max_val = df[column].max()
    df[column + '_normalized'] = (df[column] - min_val) / (max_val - min_val)
    return df

def clean_dataset(df, numeric_columns):
    cleaned_df = df.copy()
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
            cleaned_df = normalize_column(cleaned_df, col)
    cleaned_df = cleaned_df.dropna()
    return cleaned_df

if __name__ == "__main__":
    sample_data = {
        'feature1': np.random.normal(100, 15, 200),
        'feature2': np.random.exponential(50, 200),
        'category': np.random.choice(['A', 'B', 'C'], 200)
    }
    df = pd.DataFrame(sample_data)
    df.loc[10, 'feature1'] = 500
    df.loc[20, 'feature2'] = 1000
    
    numeric_cols = ['feature1', 'feature2']
    result_df = clean_dataset(df, numeric_cols)
    print(f"Original shape: {df.shape}")
    print(f"Cleaned shape: {result_df.shape}")
    print(result_df.head())import pandas as pd

def clean_dataframe(df):
    """
    Clean a pandas DataFrame by removing null values and duplicate rows.
    
    Args:
        df (pd.DataFrame): Input DataFrame to be cleaned.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    
    cleaned_df = df.copy()
    
    cleaned_df = cleaned_df.dropna()
    
    cleaned_df = cleaned_df.drop_duplicates()
    
    cleaned_df = cleaned_df.reset_index(drop=True)
    
    return cleaned_df
import numpy as np
import pandas as pd

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

def calculate_basic_stats(df, column):
    """
    Calculate basic statistics for a DataFrame column.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to analyze
    
    Returns:
    dict: Dictionary containing statistical measures
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    stats = {
        'mean': df[column].mean(),
        'median': df[column].median(),
        'std': df[column].std(),
        'min': df[column].min(),
        'max': df[column].max(),
        'count': df[column].count(),
        'missing': df[column].isnull().sum()
    }
    
    return stats

def example_usage():
    """
    Example usage of the data cleaning functions.
    """
    np.random.seed(42)
    
    data = {
        'values': np.concatenate([
            np.random.normal(100, 15, 95),
            np.array([300, 350, 400, -50, -100])
        ])
    }
    
    df = pd.DataFrame(data)
    
    print("Original DataFrame shape:", df.shape)
    print("Original statistics:", calculate_basic_stats(df, 'values'))
    
    cleaned_df = remove_outliers_iqr(df, 'values')
    
    print("\nCleaned DataFrame shape:", cleaned_df.shape)
    print("Cleaned statistics:", calculate_basic_stats(cleaned_df, 'values'))
    
    return cleaned_df

if __name__ == "__main__":
    cleaned_data = example_usage()import csv
import sys
from typing import List, Dict, Any

def read_csv(filepath: str) -> List[Dict[str, Any]]:
    """Read CSV file and return list of dictionaries."""
    data = []
    try:
        with open(filepath, 'r', newline='', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                data.append(row)
    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        sys.exit(1)
    return data

def remove_duplicates(data: List[Dict[str, Any]], key: str) -> List[Dict[str, Any]]:
    """Remove duplicate rows based on a specified key column."""
    seen = set()
    unique_data = []
    for row in data:
        if key in row:
            value = row[key]
            if value not in seen:
                seen.add(value)
                unique_data.append(row)
    return unique_data

def convert_column_types(data: List[Dict[str, Any]], conversions: Dict[str, type]) -> List[Dict[str, Any]]:
    """Convert specified columns to given data types."""
    converted_data = []
    for row in data:
        new_row = row.copy()
        for col, target_type in conversions.items():
            if col in new_row:
                try:
                    if target_type == int:
                        new_row[col] = int(float(new_row[col]))
                    elif target_type == float:
                        new_row[col] = float(new_row[col])
                    elif target_type == bool:
                        new_row[col] = new_row[col].lower() in ('true', '1', 'yes', 'y')
                except (ValueError, TypeError):
                    new_row[col] = None
        converted_data.append(new_row)
    return converted_data

def write_csv(data: List[Dict[str, Any]], filepath: str) -> None:
    """Write data to CSV file."""
    if not data:
        print("No data to write.")
        return
    fieldnames = data[0].keys()
    try:
        with open(filepath, 'w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)
        print(f"Successfully wrote {len(data)} rows to '{filepath}'.")
    except Exception as e:
        print(f"Error writing CSV: {e}")

def clean_csv(input_file: str, output_file: str, unique_key: str, type_conversions: Dict[str, type]) -> None:
    """Main function to clean CSV data."""
    print(f"Reading data from '{input_file}'...")
    data = read_csv(input_file)
    print(f"Original rows: {len(data)}")
    
    if unique_key:
        data = remove_duplicates(data, unique_key)
        print(f"After duplicate removal: {len(data)} rows")
    
    if type_conversions:
        data = convert_column_types(data, type_conversions)
        print("Column type conversions applied.")
    
    write_csv(data, output_file)

if __name__ == "__main__":
    input_csv = "input_data.csv"
    output_csv = "cleaned_data.csv"
    key_column = "id"
    conversions = {
        "age": int,
        "score": float,
        "active": bool
    }
    clean_csv(input_csv, output_csv, key_column, conversions)
import pandas as pd
import numpy as np

def clean_dataframe(df):
    """
    Clean a pandas DataFrame by removing duplicate rows and
    handling missing values in numeric columns.
    """
    # Remove duplicate rows
    df_cleaned = df.drop_duplicates()
    
    # Fill missing values in numeric columns with column median
    numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
    df_cleaned[numeric_cols] = df_cleaned[numeric_cols].fillna(
        df_cleaned[numeric_cols].median()
    )
    
    # For non-numeric columns, fill with mode (most frequent value)
    non_numeric_cols = df_cleaned.select_dtypes(exclude=[np.number]).columns
    for col in non_numeric_cols:
        if df_cleaned[col].isnull().any():
            mode_value = df_cleaned[col].mode()
            if not mode_value.empty:
                df_cleaned[col] = df_cleaned[col].fillna(mode_value.iloc[0])
    
    return df_cleaned

def validate_dataframe(df):
    """
    Validate that DataFrame has no null values after cleaning.
    """
    null_counts = df.isnull().sum()
    total_nulls = null_counts.sum()
    
    if total_nulls == 0:
        return True, "Data validation passed: No null values found."
    else:
        error_msg = f"Data validation failed: {total_nulls} null values remain."
        return False, error_msg

def process_data_file(input_path, output_path=None):
    """
    Main function to read, clean, and save data.
    """
    try:
        # Read data
        if input_path.endswith('.csv'):
            df = pd.read_csv(input_path)
        elif input_path.endswith('.xlsx'):
            df = pd.read_excel(input_path)
        else:
            raise ValueError("Unsupported file format. Use .csv or .xlsx")
        
        # Clean data
        df_cleaned = clean_dataframe(df)
        
        # Validate cleaning
        is_valid, message = validate_dataframe(df_cleaned)
        print(message)
        
        # Save cleaned data
        if output_path:
            if output_path.endswith('.csv'):
                df_cleaned.to_csv(output_path, index=False)
            elif output_path.endswith('.xlsx'):
                df_cleaned.to_excel(output_path, index=False)
            print(f"Cleaned data saved to: {output_path}")
        
        return df_cleaned, is_valid
        
    except FileNotFoundError:
        print(f"Error: File not found at {input_path}")
        return None, False
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        return None, False

if __name__ == "__main__":
    # Example usage
    input_file = "raw_data.csv"
    output_file = "cleaned_data.csv"
    
    cleaned_df, success = process_data_file(input_file, output_file)
    
    if success and cleaned_df is not None:
        print(f"Data cleaning completed successfully.")
        print(f"Original shape: {pd.read_csv(input_file).shape}")
        print(f"Cleaned shape: {cleaned_df.shape}")