
import pandas as pd
import numpy as np
from scipy import stats

def remove_outliers_iqr(df, columns):
    cleaned_df = df.copy()
    for col in columns:
        Q1 = cleaned_df[col].quantile(0.25)
        Q3 = cleaned_df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        cleaned_df = cleaned_df[(cleaned_df[col] >= lower_bound) & (cleaned_df[col] <= upper_bound)]
    return cleaned_df

def normalize_minmax(df, columns):
    normalized_df = df.copy()
    for col in columns:
        min_val = normalized_df[col].min()
        max_val = normalized_df[col].max()
        if max_val != min_val:
            normalized_df[col] = (normalized_df[col] - min_val) / (max_val - min_val)
        else:
            normalized_df[col] = 0
    return normalized_df

def clean_dataset(file_path, numeric_columns):
    try:
        df = pd.read_csv(file_path)
        df_cleaned = remove_outliers_iqr(df, numeric_columns)
        df_normalized = normalize_minmax(df_cleaned, numeric_columns)
        return df_normalized
    except Exception as e:
        print(f"Error processing file: {e}")
        return None

if __name__ == "__main__":
    cleaned_data = clean_dataset('sample_data.csv', ['age', 'income', 'score'])
    if cleaned_data is not None:
        cleaned_data.to_csv('cleaned_data.csv', index=False)
        print("Data cleaning completed successfully.")