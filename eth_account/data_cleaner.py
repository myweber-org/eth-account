
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

def clean_dataset(file_path, numeric_columns):
    df = pd.read_csv(file_path)
    
    for col in numeric_columns:
        if col in df.columns:
            df = remove_outliers_iqr(df, col)
            df = normalize_minmax(df, col)
    
    df = df.dropna()
    return df

def calculate_statistics(df, column):
    stats_dict = {
        'mean': df[column].mean(),
        'median': df[column].median(),
        'std': df[column].std(),
        'skewness': stats.skew(df[column].dropna()),
        'kurtosis': stats.kurtosis(df[column].dropna())
    }
    return stats_dict

if __name__ == "__main__":
    data_path = 'raw_data.csv'
    numeric_cols = ['price', 'quantity', 'rating']
    
    cleaned_data = clean_dataset(data_path, numeric_cols)
    cleaned_data.to_csv('cleaned_data.csv', index=False)
    
    for col in numeric_cols:
        if col in cleaned_data.columns:
            stats_result = calculate_statistics(cleaned_data, col)
            print(f"Statistics for {col}: {stats_result}")