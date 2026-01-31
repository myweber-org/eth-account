
import csv
import os
from typing import List, Dict, Any

def read_csv_file(file_path: str) -> List[Dict[str, Any]]:
    """Read a CSV file and return a list of dictionaries."""
    data = []
    try:
        with open(file_path, mode='r', newline='', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                data.append(row)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
    except Exception as e:
        print(f"Error reading file: {e}")
    return data

def remove_empty_rows(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove rows where all values are empty strings."""
    cleaned_data = []
    for row in data:
        if any(value.strip() != '' for value in row.values()):
            cleaned_data.append(row)
    return cleaned_data

def standardize_column_names(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Standardize column names to lowercase with underscores."""
    if not data:
        return data
    original_columns = list(data[0].keys())
    standardized_columns = [col.lower().replace(' ', '_') for col in original_columns]
    standardized_data = []
    for row in data:
        new_row = {}
        for old_col, new_col in zip(original_columns, standardized_columns):
            new_row[new_col] = row.get(old_col, '')
        standardized_data.append(new_row)
    return standardized_data

def clean_numeric_column(data: List[Dict[str, Any]], column_name: str) -> List[Dict[str, Any]]:
    """Clean a numeric column by removing non-numeric characters and converting to float."""
    cleaned_data = []
    for row in data:
        new_row = row.copy()
        if column_name in new_row:
            value = str(new_row[column_name])
            numeric_value = ''.join(ch for ch in value if ch.isdigit() or ch == '.')
            try:
                new_row[column_name] = float(numeric_value) if numeric_value else 0.0
            except ValueError:
                new_row[column_name] = 0.0
        cleaned_data.append(new_row)
    return cleaned_data

def write_csv_file(data: List[Dict[str, Any]], file_path: str) -> bool:
    """Write data to a CSV file."""
    if not data:
        print("No data to write.")
        return False
    try:
        with open(file_path, mode='w', newline='', encoding='utf-8') as file:
            fieldnames = list(data[0].keys())
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)
        return True
    except Exception as e:
        print(f"Error writing file: {e}")
        return False

def process_csv(input_path: str, output_path: str) -> None:
    """Main function to process a CSV file through cleaning steps."""
    print(f"Processing {input_path}")
    data = read_csv_file(input_path)
    if not data:
        print("No data loaded. Exiting.")
        return
    
    print(f"Loaded {len(data)} rows.")
    data = remove_empty_rows(data)
    print(f"After removing empty rows: {len(data)} rows.")
    data = standardize_column_names(data)
    data = clean_numeric_column(data, 'price')
    
    if write_csv_file(data, output_path):
        print(f"Cleaned data written to {output_path}")
    else:
        print("Failed to write output file.")

if __name__ == "__main__":
    input_file = "raw_data.csv"
    output_file = "cleaned_data.csv"
    process_csv(input_file, output_file)import numpy as np
import pandas as pd

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the IQR method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to clean
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
    return filtered_df

def calculate_summary_statistics(df, column):
    """
    Calculate summary statistics for a column after outlier removal.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to analyze
    
    Returns:
    dict: Dictionary containing summary statistics
    """
    stats = {
        'mean': df[column].mean(),
        'median': df[column].median(),
        'std': df[column].std(),
        'min': df[column].min(),
        'max': df[column].max(),
        'count': df[column].count()
    }
    
    return stats

def process_dataframe(df, column):
    """
    Main function to process DataFrame by removing outliers and calculating statistics.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to process
    
    Returns:
    tuple: (cleaned_df, original_stats, cleaned_stats)
    """
    original_stats = calculate_summary_statistics(df, column)
    cleaned_df = remove_outliers_iqr(df, column)
    cleaned_stats = calculate_summary_statistics(cleaned_df, column)
    
    return cleaned_df, original_stats, cleaned_stats