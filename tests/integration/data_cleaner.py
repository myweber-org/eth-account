import numpy as np
import pandas as pd

def remove_outliers_iqr(data, column, factor=1.5):
    """
    Remove outliers using the Interquartile Range method.
    """
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - factor * IQR
    upper_bound = Q3 + factor * IQR
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data

def normalize_minmax(data, column):
    """
    Normalize data using Min-Max scaling.
    """
    min_val = data[column].min()
    max_val = data[column].max()
    if max_val - min_val == 0:
        return data[column]
    normalized = (data[column] - min_val) / (max_val - min_val)
    return normalized

def standardize_zscore(data, column):
    """
    Standardize data using Z-score normalization.
    """
    mean_val = data[column].mean()
    std_val = data[column].std()
    if std_val == 0:
        return data[column]
    standardized = (data[column] - mean_val) / std_val
    return standardized

def clean_dataset(df, numeric_columns, outlier_factor=1.5, normalization_method='minmax'):
    """
    Clean dataset by removing outliers and applying normalization.
    """
    cleaned_df = df.copy()
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, col, outlier_factor)
            if normalization_method == 'minmax':
                cleaned_df[col] = normalize_minmax(cleaned_df, col)
            elif normalization_method == 'zscore':
                cleaned_df[col] = standardize_zscore(cleaned_df, col)
    return cleaned_df

def main():
    sample_data = {
        'A': np.random.normal(100, 15, 100),
        'B': np.random.exponential(50, 100),
        'C': np.random.uniform(0, 1, 100)
    }
    df = pd.DataFrame(sample_data)
    print("Original dataset shape:", df.shape)
    print("Original statistics:")
    print(df.describe())
    
    cleaned_df = clean_dataset(df, ['A', 'B', 'C'], outlier_factor=1.5, normalization_method='minmax')
    print("\nCleaned dataset shape:", cleaned_df.shape)
    print("Cleaned statistics:")
    print(cleaned_df.describe())

if __name__ == "__main__":
    main()import csv
import os
from typing import List, Dict, Any

def read_csv(filepath: str) -> List[Dict[str, Any]]:
    """Read a CSV file and return a list of dictionaries."""
    data = []
    try:
        with open(filepath, 'r', newline='', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                data.append(row)
    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found.")
    except Exception as e:
        print(f"Error reading CSV: {e}")
    return data

def write_csv(filepath: str, data: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    """Write a list of dictionaries to a CSV file."""
    try:
        with open(filepath, 'w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)
        print(f"Data successfully written to '{filepath}'.")
    except Exception as e:
        print(f"Error writing CSV: {e}")

def clean_numeric(value: str) -> float:
    """Clean a numeric string by removing non-numeric characters."""
    if not value:
        return 0.0
    cleaned = ''.join(ch for ch in value if ch.isdigit() or ch == '.')
    try:
        return float(cleaned) if cleaned else 0.0
    except ValueError:
        return 0.0

def remove_duplicates(data: List[Dict[str, Any]], key: str) -> List[Dict[str, Any]]:
    """Remove duplicate rows based on a specified key."""
    seen = set()
    unique_data = []
    for row in data:
        if row[key] not in seen:
            seen.add(row[key])
            unique_data.append(row)
    return unique_data

def filter_by_threshold(data: List[Dict[str, Any]], column: str, threshold: float) -> List[Dict[str, Any]]:
    """Filter rows where the numeric value in a column is above a threshold."""
    filtered = []
    for row in data:
        try:
            value = float(row[column])
            if value > threshold:
                filtered.append(row)
        except (ValueError, KeyError):
            continue
    return filtered

def main():
    """Example usage of the data cleaning functions."""
    input_file = "input_data.csv"
    output_file = "cleaned_data.csv"
    
    data = read_csv(input_file)
    if not data:
        print("No data to process.")
        return
    
    print(f"Original data count: {len(data)}")
    
    # Clean numeric columns
    for row in data:
        if 'price' in row:
            row['price'] = clean_numeric(row['price'])
    
    # Remove duplicates by 'id' column
    if 'id' in data[0]:
        data = remove_duplicates(data, 'id')
    
    # Filter rows where price > 100
    if 'price' in data[0]:
        data = filter_by_threshold(data, 'price', 100.0)
    
    print(f"Cleaned data count: {len(data)}")
    
    # Write cleaned data to new CSV
    if data:
        fieldnames = list(data[0].keys())
        write_csv(output_file, data, fieldnames)

if __name__ == "__main__":
    main()