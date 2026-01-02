
import pandas as pd
import numpy as np
from scipy import stats

def remove_outliers_iqr(dataframe, columns):
    cleaned_df = dataframe.copy()
    for col in columns:
        if col in cleaned_df.columns:
            Q1 = cleaned_df[col].quantile(0.25)
            Q3 = cleaned_df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            cleaned_df = cleaned_df[(cleaned_df[col] >= lower_bound) & (cleaned_df[col] <= upper_bound)]
    return cleaned_df

def normalize_data_minmax(dataframe, columns):
    normalized_df = dataframe.copy()
    for col in columns:
        if col in normalized_df.columns:
            min_val = normalized_df[col].min()
            max_val = normalized_df[col].max()
            if max_val != min_val:
                normalized_df[col] = (normalized_df[col] - min_val) / (max_val - min_val)
    return normalized_df

def clean_dataset(input_path, output_path, outlier_cols=None, normalize_cols=None):
    try:
        df = pd.read_csv(input_path)
        
        if outlier_cols:
            df = remove_outliers_iqr(df, outlier_cols)
        
        if normalize_cols:
            df = normalize_data_minmax(df, normalize_cols)
        
        df.to_csv(output_path, index=False)
        print(f"Cleaned data saved to {output_path}")
        return True
        
    except Exception as e:
        print(f"Error during data cleaning: {e}")
        return False

if __name__ == "__main__":
    input_file = "raw_data.csv"
    output_file = "cleaned_data.csv"
    
    outlier_columns = ['age', 'income', 'score']
    normalize_columns = ['income', 'score']
    
    clean_dataset(input_file, output_file, outlier_columns, normalize_columns)