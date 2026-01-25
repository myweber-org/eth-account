import csv
import os
from typing import List, Dict, Any, Optional

def read_csv_file(file_path: str) -> List[Dict[str, Any]]:
    """Read a CSV file and return a list of dictionaries."""
    data = []
    try:
        with open(file_path, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                data.append(row)
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    except Exception as e:
        print(f"Error reading CSV file: {e}")
    return data

def clean_numeric_field(record: Dict[str, Any], field_name: str, default_value: float = 0.0) -> None:
    """Clean a numeric field in a record, converting it to float."""
    if field_name in record:
        try:
            value = record[field_name].strip()
            if value:
                record[field_name] = float(value)
            else:
                record[field_name] = default_value
        except (ValueError, AttributeError):
            record[field_name] = default_value

def remove_duplicates(data: List[Dict[str, Any]], key_field: str) -> List[Dict[str, Any]]:
    """Remove duplicate records based on a key field."""
    seen = set()
    unique_data = []
    for record in data:
        key = record.get(key_field)
        if key not in seen:
            seen.add(key)
            unique_data.append(record)
    return unique_data

def filter_records(data: List[Dict[str, Any]], filter_func) -> List[Dict[str, Any]]:
    """Filter records using a custom filter function."""
    return [record for record in data if filter_func(record)]

def write_csv_file(data: List[Dict[str, Any]], file_path: str, fieldnames: Optional[List[str]] = None) -> bool:
    """Write data to a CSV file."""
    if not data:
        print("No data to write.")
        return False
    
    if fieldnames is None:
        fieldnames = list(data[0].keys())
    
    try:
        with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)
        return True
    except Exception as e:
        print(f"Error writing CSV file: {e}")
        return False

def process_csv(input_file: str, output_file: str, key_field: str = "id") -> None:
    """Main function to process a CSV file: read, clean, deduplicate, and write."""
    print(f"Processing {input_file}...")
    
    data = read_csv_file(input_file)
    if not data:
        print("No data loaded.")
        return
    
    print(f"Loaded {len(data)} records.")
    
    for record in data:
        clean_numeric_field(record, "price")
        clean_numeric_field(record, "quantity")
    
    data = remove_duplicates(data, key_field)
    print(f"After removing duplicates: {len(data)} records.")
    
    def valid_price_filter(record):
        price = record.get("price", 0)
        return price > 0
    
    data = filter_records(data, valid_price_filter)
    print(f"After filtering invalid prices: {len(data)} records.")
    
    if write_csv_file(data, output_file):
        print(f"Cleaned data written to {output_file}")import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(dataframe, column):
    Q1 = dataframe[column].quantile(0.25)
    Q3 = dataframe[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    filtered_df = dataframe[(dataframe[column] >= lower_bound) & (dataframe[column] <= upper_bound)]
    return filtered_df

def normalize_minmax(dataframe, column):
    min_val = dataframe[column].min()
    max_val = dataframe[column].max()
    if max_val == min_val:
        return dataframe[column].apply(lambda x: 0.5)
    normalized = (dataframe[column] - min_val) / (max_val - min_val)
    return normalized

def standardize_zscore(dataframe, column):
    mean_val = dataframe[column].mean()
    std_val = dataframe[column].std()
    if std_val == 0:
        return dataframe[column].apply(lambda x: 0)
    standardized = (dataframe[column] - mean_val) / std_val
    return standardized

def handle_missing_mean(dataframe, column):
    mean_val = dataframe[column].mean()
    filled_series = dataframe[column].fillna(mean_val)
    return filled_series

def validate_numeric_range(dataframe, column, min_val, max_val):
    invalid_mask = (dataframe[column] < min_val) | (dataframe[column] > max_val)
    validation_result = ~invalid_mask
    return validation_result