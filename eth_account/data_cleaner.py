import pandas as pd
import numpy as np
from scipy import stats

def detect_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    return outliers

def impute_missing_with_median(data, column):
    median_value = data[column].median()
    data[column].fillna(median_value, inplace=True)
    return data

def remove_duplicates(data, subset=None):
    if subset:
        data_cleaned = data.drop_duplicates(subset=subset)
    else:
        data_cleaned = data.drop_duplicates()
    return data_cleaned

def standardize_column(data, column):
    mean = data[column].mean()
    std = data[column].std()
    data[column] = (data[column] - mean) / std
    return data

def clean_dataset(data, numeric_columns, categorical_columns=None):
    cleaned_data = data.copy()
    
    for col in numeric_columns:
        if cleaned_data[col].isnull().any():
            cleaned_data = impute_missing_with_median(cleaned_data, col)
        outliers = detect_outliers_iqr(cleaned_data, col)
        if not outliers.empty:
            median_val = cleaned_data[col].median()
            cleaned_data.loc[outliers.index, col] = median_val
        cleaned_data = standardize_column(cleaned_data, col)
    
    if categorical_columns:
        for col in categorical_columns:
            if cleaned_data[col].isnull().any():
                mode_val = cleaned_data[col].mode()[0]
                cleaned_data[col].fillna(mode_val, inplace=True)
    
    cleaned_data = remove_duplicates(cleaned_data)
    return cleaned_data

def validate_data(data, required_columns):
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    return True

def get_data_summary(data):
    summary = {
        'total_rows': len(data),
        'total_columns': len(data.columns),
        'missing_values': data.isnull().sum().sum(),
        'duplicate_rows': data.duplicated().sum(),
        'numeric_columns': list(data.select_dtypes(include=[np.number]).columns),
        'categorical_columns': list(data.select_dtypes(include=['object']).columns)
    }
    return summary