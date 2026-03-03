import pandas as pd
import numpy as np

def detect_outliers_iqr(data, column):
    """
    Detect outliers using the Interquartile Range method.
    Returns a boolean mask where True indicates an outlier.
    """
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return (data[column] < lower_bound) | (data[column] > upper_bound)

def remove_outliers(data, column):
    """
    Remove outliers from a specified column using IQR method.
    Returns a cleaned DataFrame.
    """
    outlier_mask = detect_outliers_iqr(data, column)
    return data[~outlier_mask].copy()

def normalize_minmax(data, column):
    """
    Normalize a column using Min-Max scaling to range [0, 1].
    Returns a new Series with normalized values.
    """
    min_val = data[column].min()
    max_val = data[column].max()
    if max_val == min_val:
        return pd.Series([0.5] * len(data), index=data.index)
    return (data[column] - min_val) / (max_val - min_val)

def clean_dataset(df, numeric_columns):
    """
    Main cleaning function: removes outliers and normalizes numeric columns.
    Returns a cleaned DataFrame and a dictionary of outlier counts per column.
    """
    cleaned_df = df.copy()
    outlier_info = {}
    
    for col in numeric_columns:
        if col in cleaned_df.columns:
            outliers = detect_outliers_iqr(cleaned_df, col)
            outlier_info[col] = outliers.sum()
            cleaned_df = remove_outliers(cleaned_df, col)
            cleaned_df[f'{col}_normalized'] = normalize_minmax(cleaned_df, col)
    
    return cleaned_df, outlier_info

def save_cleaned_data(df, output_path):
    """
    Save cleaned DataFrame to a CSV file.
    """
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to {output_path}")