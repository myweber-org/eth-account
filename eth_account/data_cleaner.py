
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

def zscore_normalize(df, column):
    mean_val = df[column].mean()
    std_val = df[column].std()
    df[column + '_normalized'] = (df[column] - mean_val) / std_val
    return df

def minmax_normalize(df, column):
    min_val = df[column].min()
    max_val = df[column].max()
    df[column + '_normalized'] = (df[column] - min_val) / (max_val - min_val)
    return df

def handle_missing_values(df, strategy='mean'):
    if strategy == 'mean':
        return df.fillna(df.mean())
    elif strategy == 'median':
        return df.fillna(df.median())
    elif strategy == 'mode':
        return df.fillna(df.mode().iloc[0])
    else:
        return df.dropna()

def detect_anomalies_zscore(df, column, threshold=3):
    z_scores = np.abs(stats.zscore(df[column]))
    anomalies = df[z_scores > threshold]
    return anomalies

def clean_dataset(df, numeric_columns):
    cleaned_df = df.copy()
    for col in numeric_columns:
        cleaned_df = remove_outliers_iqr(cleaned_df, col)
        cleaned_df = handle_missing_values(cleaned_df)
    return cleaned_df