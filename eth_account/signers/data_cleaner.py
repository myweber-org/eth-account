
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def normalize_column(df, column):
    min_val = df[column].min()
    max_val = df[column].max()
    if max_val != min_val:
        df[column] = (df[column] - min_val) / (max_val - min_val)
    return df

def clean_dataset(df, numeric_columns):
    cleaned_df = df.copy()
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
            cleaned_df = normalize_column(cleaned_df, col)
    cleaned_df = cleaned_df.dropna()
    return cleaned_df

if __name__ == "__main__":
    sample_data = {
        'feature1': np.random.normal(100, 15, 200),
        'feature2': np.random.exponential(50, 200),
        'category': np.random.choice(['A', 'B', 'C'], 200)
    }
    sample_df = pd.DataFrame(sample_data)
    sample_df.loc[10, 'feature1'] = 500
    sample_df.loc[20, 'feature2'] = 1000
    
    numeric_cols = ['feature1', 'feature2']
    result_df = clean_dataset(sample_df, numeric_cols)
    
    print(f"Original shape: {sample_df.shape}")
    print(f"Cleaned shape: {result_df.shape}")
    print(f"Feature1 range: [{result_df['feature1'].min():.3f}, {result_df['feature1'].max():.3f}]")
    print(f"Feature2 range: [{result_df['feature2'].min():.3f}, {result_df['feature2'].max():.3f}]")