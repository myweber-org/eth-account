import numpy as np
import pandas as pd

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
    if max_val == min_val:
        return df[column].apply(lambda x: 0.0)
    return df[column].apply(lambda x: (x - min_val) / (max_val - min_val))

def clean_dataset(df, numeric_columns):
    cleaned_df = df.copy()
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
            cleaned_df[col] = normalize_minmax(cleaned_df, col)
    return cleaned_df.reset_index(drop=True)

def validate_cleaning(df_before, df_after, column):
    if column not in df_before.columns or column not in df_after.columns:
        return None
    stats = {
        'original_mean': df_before[column].mean(),
        'cleaned_mean': df_after[column].mean(),
        'original_std': df_before[column].std(),
        'cleaned_std': df_after[column].std(),
        'rows_removed': len(df_before) - len(df_after)
    }
    return stats

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'feature_a': np.random.normal(100, 15, 200),
        'feature_b': np.random.exponential(50, 200)
    })
    cleaned = clean_dataset(sample_data, ['feature_a', 'feature_b'])
    validation = validate_cleaning(sample_data, cleaned, 'feature_a')
    print(f"Original rows: {len(sample_data)}")
    print(f"Cleaned rows: {len(cleaned)}")
    print(f"Validation stats: {validation}")