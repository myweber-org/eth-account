
import csv
import os

def load_csv(file_path):
    """Load CSV file and return data as list of dictionaries."""
    data = []
    try:
        with open(file_path, 'r', newline='', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                data.append(row)
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    except Exception as e:
        print(f"Error loading CSV: {e}")
    return data

def clean_numeric_fields(data, fields):
    """Remove non-numeric characters from specified fields."""
    cleaned_data = []
    for row in data:
        cleaned_row = row.copy()
        for field in fields:
            if field in cleaned_row:
                value = cleaned_row[field]
                if isinstance(value, str):
                    cleaned_row[field] = ''.join(char for char in value if char.isdigit() or char == '.')
        cleaned_data.append(cleaned_row)
    return cleaned_data

def remove_empty_rows(data, required_fields):
    """Remove rows where any required field is empty."""
    filtered_data = []
    for row in data:
        if all(row.get(field) not in [None, ''] for field in required_fields):
            filtered_data.append(row)
    return filtered_data

def save_csv(data, file_path):
    """Save data to CSV file."""
    if not data:
        print("No data to save.")
        return False
    
    try:
        with open(file_path, 'w', newline='', encoding='utf-8') as file:
            fieldnames = data[0].keys()
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)
        return True
    except Exception as e:
        print(f"Error saving CSV: {e}")
        return False

def process_csv(input_path, output_path, numeric_fields=None, required_fields=None):
    """Main function to process CSV file."""
    if numeric_fields is None:
        numeric_fields = []
    if required_fields is None:
        required_fields = []
    
    data = load_csv(input_path)
    if not data:
        return False
    
    data = clean_numeric_fields(data, numeric_fields)
    data = remove_empty_rows(data, required_fields)
    
    return save_csv(data, output_path)import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column, threshold=1.5):
    """
    Remove outliers using IQR method.
    """
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data

def normalize_minmax(data, column):
    """
    Normalize data using min-max scaling.
    """
    min_val = data[column].min()
    max_val = data[column].max()
    if max_val - min_val == 0:
        return data[column]
    normalized = (data[column] - min_val) / (max_val - min_val)
    return normalized

def standardize_zscore(data, column):
    """
    Standardize data using z-score normalization.
    """
    mean_val = data[column].mean()
    std_val = data[column].std()
    if std_val == 0:
        return data[column]
    standardized = (data[column] - mean_val) / std_val
    return standardized

def clean_dataset(df, numeric_columns, outlier_threshold=1.5, normalization_method='standardize'):
    """
    Clean dataset by removing outliers and applying normalization.
    """
    cleaned_df = df.copy()
    
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, col, outlier_threshold)
            
            if normalization_method == 'minmax':
                cleaned_df[col] = normalize_minmax(cleaned_df, col)
            elif normalization_method == 'standardize':
                cleaned_df[col] = standardize_zscore(cleaned_df, col)
    
    return cleaned_df

def validate_data(df, required_columns, allow_nan=False):
    """
    Validate dataset structure and content.
    """
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    if not allow_nan:
        nan_count = df.isnull().sum().sum()
        if nan_count > 0:
            df_cleaned = df.dropna()
            print(f"Removed {nan_count} NaN values")
            return df_cleaned
    
    return df

def generate_summary(df):
    """
    Generate statistical summary of the dataset.
    """
    summary = {
        'rows': len(df),
        'columns': len(df.columns),
        'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
        'categorical_columns': df.select_dtypes(include=['object']).columns.tolist(),
        'missing_values': df.isnull().sum().to_dict(),
        'basic_stats': df.describe().to_dict()
    }
    return summary