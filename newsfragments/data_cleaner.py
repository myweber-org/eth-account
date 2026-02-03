import pandas as pd
import numpy as np
from scipy import stats

def load_and_clean_data(filepath):
    df = pd.read_csv(filepath)
    print(f"Original shape: {df.shape}")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        z_scores = np.abs(stats.zscore(df[col].dropna()))
        outliers = df[col].index[z_scores > 3]
        df.loc[outliers, col] = np.nan
        print(f"Removed {len(outliers)} outliers from {col}")
    
    df_cleaned = df.dropna()
    print(f"Cleaned shape: {df_cleaned.shape}")
    
    for col in numeric_cols:
        if col in df_cleaned.columns:
            col_min = df_cleaned[col].min()
            col_max = df_cleaned[col].max()
            if col_max > col_min:
                df_cleaned[col] = (df_cleaned[col] - col_min) / (col_max - col_min)
    
    return df_cleaned

if __name__ == "__main__":
    cleaned_df = load_and_clean_data("sample_data.csv")
    cleaned_df.to_csv("cleaned_data.csv", index=False)
    print("Data cleaning complete. Saved to cleaned_data.csv")