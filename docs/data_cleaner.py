
import pandas as pd
import numpy as np
from scipy import stats

def remove_outliers_iqr(df, columns):
    df_clean = df.copy()
    for col in columns:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
    return df_clean

def normalize_data(df, columns, method='minmax'):
    df_norm = df.copy()
    for col in columns:
        if method == 'minmax':
            df_norm[col] = (df_norm[col] - df_norm[col].min()) / (df_norm[col].max() - df_norm[col].min())
        elif method == 'zscore':
            df_norm[col] = stats.zscore(df_norm[col])
        elif method == 'log':
            df_norm[col] = np.log1p(df_norm[col])
    return df_norm

def clean_dataset(file_path, numeric_columns, outlier_removal=True, normalization='minmax'):
    df = pd.read_csv(file_path)
    
    if outlier_removal:
        df = remove_outliers_iqr(df, numeric_columns)
    
    if normalization:
        df = normalize_data(df, numeric_columns, method=normalization)
    
    return df

if __name__ == "__main__":
    cleaned_df = clean_dataset(
        'sample_data.csv',
        ['age', 'income', 'score'],
        outlier_removal=True,
        normalization='zscore'
    )
    print(f"Cleaned dataset shape: {cleaned_df.shape}")
    print(cleaned_df.head())