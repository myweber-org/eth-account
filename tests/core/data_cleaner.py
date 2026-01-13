import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]

def normalize_minmax(data, column):
    min_val = data[column].min()
    max_val = data[column].max()
    data[column + '_normalized'] = (data[column] - min_val) / (max_val - min_val)
    return data

def z_score_normalize(data, column):
    mean_val = data[column].mean()
    std_val = data[column].std()
    data[column + '_zscore'] = (data[column] - mean_val) / std_val
    return data

def clean_dataset(df, numeric_columns, method='iqr'):
    cleaned_df = df.copy()
    
    for col in numeric_columns:
        if method == 'iqr':
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
        elif method == 'zscore':
            z_scores = np.abs(stats.zscore(cleaned_df[col]))
            cleaned_df = cleaned_df[z_scores < 3]
    
    return cleaned_df.reset_index(drop=True)

def process_data_pipeline(df, numeric_cols, outlier_method='iqr', normalize_method='minmax'):
    df_cleaned = clean_dataset(df, numeric_cols, outlier_method)
    
    for col in numeric_cols:
        if normalize_method == 'minmax':
            df_cleaned = normalize_minmax(df_cleaned, col)
        elif normalize_method == 'zscore':
            df_cleaned = z_score_normalize(df_cleaned, col)
    
    return df_cleaned