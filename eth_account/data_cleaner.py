
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
                if std_val > 0:
                    normalized_df[col] = (normalized_df[col] - mean_val) / std_val
    return normalized_df

def clean_dataset(df, numeric_columns, outlier_threshold=1.5, normalize_method='minmax'):
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame")
    
    df_clean = df.copy()
    df_clean = df_clean.dropna(subset=numeric_columns)
    
    df_no_outliers = remove_outliers_iqr(df_clean, numeric_columns)
    
    df_normalized = normalize_data(df_no_outliers, numeric_columns, method=normalize_method)
    
    return df_normalized

def validate_data(df, numeric_columns):
    validation_report = {}
    for col in numeric_columns:
        if col in df.columns:
            validation_report[col] = {
                'mean': df[col].mean(),
                'std': df[col].std(),
                'min': df[col].min(),
                'max': df[col].max(),
                'missing': df[col].isnull().sum(),
                'zeros': (df[col] == 0).sum()
            }
    return validation_report

if __name__ == "__main__":
    sample_data = {
        'feature_a': np.random.normal(100, 15, 1000),
        'feature_b': np.random.exponential(50, 1000),
        'feature_c': np.random.uniform(0, 200, 1000)
    }
    
    df = pd.DataFrame(sample_data)
    df.loc[np.random.choice(df.index, 50), 'feature_a'] = np.nan
    
    numeric_cols = ['feature_a', 'feature_b', 'feature_c']
    
    cleaned_df = clean_dataset(df, numeric_cols)
    
    validation = validate_data(cleaned_df, numeric_cols)
    
    print(f"Original shape: {df.shape}")
    print(f"Cleaned shape: {cleaned_df.shape}")
    print("\nValidation summary:")
    for col, stats in validation.items():
        print(f"{col}: mean={stats['mean']:.2f}, std={stats['std']:.2f}")