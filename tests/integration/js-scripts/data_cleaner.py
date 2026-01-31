import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column, factor=1.5):
    """
    Remove outliers using IQR method
    """
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - factor * iqr
    upper_bound = q3 + factor * iqr
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data

def z_score_normalization(data, column):
    """
    Apply z-score normalization to specified column
    """
    mean = data[column].mean()
    std = data[column].std()
    
    if std > 0:
        data[f'{column}_normalized'] = (data[column] - mean) / std
    else:
        data[f'{column}_normalized'] = 0
    
    return data

def min_max_normalization(data, column, new_min=0, new_max=1):
    """
    Apply min-max normalization to specified column
    """
    old_min = data[column].min()
    old_max = data[column].max()
    
    if old_max - old_min > 0:
        data[f'{column}_scaled'] = ((data[column] - old_min) / (old_max - old_min)) * (new_max - new_min) + new_min
    else:
        data[f'{column}_scaled'] = new_min
    
    return data

def clean_dataset(df, numeric_columns):
    """
    Main cleaning function that processes multiple numeric columns
    """
    cleaned_df = df.copy()
    
    for col in numeric_columns:
        if col in cleaned_df.columns:
            # Remove outliers
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
            
            # Apply both normalizations
            cleaned_df = z_score_normalization(cleaned_df, col)
            cleaned_df = min_max_normalization(cleaned_df, col)
    
    return cleaned_df

def validate_data(df, required_columns):
    """
    Validate that required columns exist and have no null values
    """
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    null_counts = df[required_columns].isnull().sum()
    columns_with_nulls = null_counts[null_counts > 0].index.tolist()
    
    return {
        'is_valid': len(missing_columns) == 0 and len(columns_with_nulls) == 0,
        'missing_columns': missing_columns,
        'columns_with_nulls': columns_with_nulls
    }import csv
import re
from typing import List, Dict, Optional

def clean_string(value: str) -> str:
    """Remove extra whitespace and normalize string."""
    if not isinstance(value, str):
        return str(value)
    cleaned = re.sub(r'\s+', ' ', value.strip())
    return cleaned

def validate_email(email: str) -> bool:
    """Validate email format."""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))

def clean_csv_row(row: Dict[str, str]) -> Dict[str, str]:
    """Clean all string values in a CSV row dictionary."""
    cleaned_row = {}
    for key, value in row.items():
        cleaned_row[key] = clean_string(value)
    return cleaned_row

def read_and_clean_csv(filepath: str) -> List[Dict[str, str]]:
    """Read CSV file and clean all rows."""
    cleaned_data = []
    try:
        with open(filepath, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                cleaned_row = clean_csv_row(row)
                cleaned_data.append(cleaned_row)
    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found.")
    except Exception as e:
        print(f"Error reading CSV: {e}")
    
    return cleaned_data

def filter_valid_emails(data: List[Dict[str, str]], email_field: str = 'email') -> List[Dict[str, str]]:
    """Filter rows with valid email addresses."""
    valid_rows = []
    for row in data:
        if email_field in row and validate_email(row[email_field]):
            valid_rows.append(row)
    return valid_rows

def write_cleaned_csv(data: List[Dict[str, str]], output_path: str) -> None:
    """Write cleaned data to a new CSV file."""
    if not data:
        print("No data to write.")
        return
    
    fieldnames = data[0].keys()
    try:
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)
        print(f"Cleaned data written to: {output_path}")
    except Exception as e:
        print(f"Error writing CSV: {e}")