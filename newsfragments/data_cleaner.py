import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def fill_missing_with_median(df, column):
    median_value = df[column].median()
    df[column] = df[column].fillna(median_value)
    return df

def standardize_column(df, column):
    mean = df[column].mean()
    std = df[column].std()
    df[column] = (df[column] - mean) / std
    return df

def clean_dataset(df, numeric_columns):
    for col in numeric_columns:
        df = remove_outliers_iqr(df, col)
        df = fill_missing_with_median(df, col)
        df = standardize_column(df, col)
    return df