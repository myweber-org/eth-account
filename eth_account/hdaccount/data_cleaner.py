
import numpy as np
import pandas as pd

def remove_outliers_iqr(dataframe, column):
    Q1 = dataframe[column].quantile(0.25)
    Q3 = dataframe[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    filtered_df = dataframe[(dataframe[column] >= lower_bound) & (dataframe[column] <= upper_bound)]
    return filtered_df

def normalize_column(dataframe, column):
    min_val = dataframe[column].min()
    max_val = dataframe[column].max()
    if max_val - min_val == 0:
        return dataframe[column]
    normalized = (dataframe[column] - min_val) / (max_val - min_val)
    return normalized

def clean_dataset(dataframe, numeric_columns):
    cleaned_df = dataframe.copy()
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
            cleaned_df[col] = normalize_column(cleaned_df, col)
    return cleaned_df

def generate_summary(dataframe):
    summary = {
        'total_rows': len(dataframe),
        'numeric_columns': list(dataframe.select_dtypes(include=[np.number]).columns),
        'missing_values': dataframe.isnull().sum().to_dict(),
        'mean_values': dataframe.select_dtypes(include=[np.number]).mean().to_dict()
    }
    return summary

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'A': np.random.normal(100, 15, 1000),
        'B': np.random.uniform(0, 1, 1000),
        'C': np.random.exponential(2, 1000)
    })
    
    cleaned_data = clean_dataset(sample_data, ['A', 'B', 'C'])
    stats = generate_summary(cleaned_data)
    
    print(f"Original rows: {len(sample_data)}")
    print(f"Cleaned rows: {stats['total_rows']}")
    print(f"Removed outliers: {len(sample_data) - stats['total_rows']}")