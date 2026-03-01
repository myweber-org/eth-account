
import pandas as pd
import numpy as np
from scipy import stats

def remove_outliers_iqr(df, columns):
    cleaned_df = df.copy()
    for col in columns:
        if col in cleaned_df.columns:
            Q1 = cleaned_df[col].quantile(0.25)
            Q3 = cleaned_df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            cleaned_df = cleaned_df[(cleaned_df[col] >= lower_bound) & (cleaned_df[col] <= upper_bound)]
    return cleaned_df

def normalize_data(df, columns, method='minmax'):
    normalized_df = df.copy()
    for col in columns:
        if col in normalized_df.columns:
            if method == 'minmax':
                min_val = normalized_df[col].min()
                max_val = normalized_df[col].max()
                if max_val != min_val:
                    normalized_df[col] = (normalized_df[col] - min_val) / (max_val - min_val)
            elif method == 'zscore':
                mean_val = normalized_df[col].mean()
                std_val = normalized_df[col].std()
                if std_val != 0:
                    normalized_df[col] = (normalized_df[col] - mean_val) / std_val
    return normalized_df

def handle_missing_values(df, strategy='mean'):
    handled_df = df.copy()
    numeric_cols = handled_df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if handled_df[col].isnull().any():
            if strategy == 'mean':
                fill_value = handled_df[col].mean()
            elif strategy == 'median':
                fill_value = handled_df[col].median()
            elif strategy == 'mode':
                fill_value = handled_df[col].mode()[0]
            else:
                fill_value = 0
            handled_df[col].fillna(fill_value, inplace=True)
    return handled_df

def clean_dataset(df, numeric_columns, outlier_removal=True, normalization=True, missing_strategy='mean'):
    df_cleaned = df.copy()
    if outlier_removal:
        df_cleaned = remove_outliers_iqr(df_cleaned, numeric_columns)
    if missing_strategy:
        df_cleaned = handle_missing_values(df_cleaned, strategy=missing_strategy)
    if normalization:
        df_cleaned = normalize_data(df_cleaned, numeric_columns, method='minmax')
    return df_cleaned